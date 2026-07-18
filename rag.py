"""
PharmacologyGPT RAG Pipeline

This module provides:

1. Lazy initialization of retrieval and language-model resources
2. Hybrid local retrieval using Chroma and BM25
3. Question routing between local knowledge and web search
4. Context fusion
5. Grounded answer generation
"""

from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from context_fusion import build_fused_context
from hybrid_retriever import (
    create_bm25_retriever,
    load_documents_from_chroma,
    retrieve_hybrid_documents,
)
from query_rewriter import rewrite_web_query
from router import route_question
from vector_store import get_vectorstore
from web_search import search_web


# Load variables from the local .env file when available.
load_dotenv()


# ============================================================
# Lazy local retrieval resources
# ============================================================

@lru_cache(maxsize=1)
def get_vector_retriever():
    """
    Create and return the shared Chroma vector retriever.

    The retriever is created only when local retrieval is first
    requested. Later calls reuse the cached instance.
    """

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 30,
            "lambda_mult": 0.7,
        },
    )


@lru_cache(maxsize=1)
def get_bm25_retriever():
    """
    Reconstruct stored Chroma documents and create the shared
    BM25 keyword retriever.

    This operation is deferred until local retrieval is needed.
    """

    vectorstore = get_vectorstore()

    documents = load_documents_from_chroma(
        vectorstore=vectorstore,
    )

    return create_bm25_retriever(
        documents=documents,
        result_count=10,
    )


# ============================================================
# Lazy language-model resource
# ============================================================

@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """
    Create and return the shared language model.

    The OpenAI client is initialized only when answer generation
    is first requested.
    """

    return ChatOpenAI(
        model="gpt-5-nano",
        temperature=0,
    )


# ============================================================
# Hybrid retrieval
# ============================================================

def run_hybrid_retrieval(chain_input: dict):
    """
    Extract the user's question from a LangChain retrieval input
    and run hybrid vector plus BM25 retrieval.

    Parameters
    ----------
    chain_input : dict
        LangChain input containing the user's question under
        the key ``input``.

    Returns
    -------
    list
        Ranked LangChain documents.
    """

    query = chain_input.get("input", "").strip()

    if not query:
        return []

    return retrieve_hybrid_documents(
        query=query,
        vector_retriever=get_vector_retriever(),
        bm25_retriever=get_bm25_retriever(),
        vector_weight=0.5,
        bm25_weight=0.5,
        final_result_count=5,
    )


# ============================================================
# Routed context retrieval
# ============================================================

def retrieve_routed_context(question: str) -> dict:
    """
    Route the user's question and retrieve appropriate context.

    Possible routes
    ---------------
    local
        Hybrid Chroma and BM25 retrieval only.

    web
        Web search only.

    both
        Hybrid local retrieval and web search.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    route = route_question(cleaned_question)

    local_documents = []
    web_results = []
    rewritten_query = None
    query_intent = None

    # Local resources are initialized only for local or both routes.
    if route in {"local", "both"}:
        local_documents = retrieve_hybrid_documents(
            query=cleaned_question,
            vector_retriever=get_vector_retriever(),
            bm25_retriever=get_bm25_retriever(),
            vector_weight=0.5,
            bm25_weight=0.5,
            final_result_count=5,
        )

    # Web search resources are used only for web or both routes.
    if route in {"web", "both"}:
        rewritten_query, query_intent = rewrite_web_query(
            cleaned_question
        )

        web_results = search_web(
            query=rewritten_query,
            max_results=5,
        )

    fused_context = build_fused_context(
        local_documents=local_documents,
        web_results=web_results,
    )

    return {
        "route": route,
        "query_intent": (
            query_intent.value
            if query_intent is not None
            else None
        ),
        "rewritten_query": rewritten_query,
        "local_documents": local_documents,
        "web_results": web_results,
        "fused_context": fused_context,
    }


# ============================================================
# Routed answer generation
# ============================================================

def answer_routed_question(question: str) -> dict:
    """
    Route a question, retrieve relevant context, and generate
    a grounded answer.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    retrieval_result = retrieve_routed_context(
        cleaned_question
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are PharmacologyGPT, a pharmacology question-answering assistant.

Answer the user's question using only the provided context.

Rules:
1. Do not use information outside the provided context.
2. Clearly distinguish textbook knowledge from current web information.
3. If the context does not contain enough information, say so.
4. Do not invent drug facts, warnings, approvals, dosages, or medical advice.
5. When web information is used, mention that it comes from current web sources.
6. Give a clear and concise answer.

Context:
{context}
""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    answer_chain = answer_prompt | get_llm()

    response = answer_chain.invoke(
        {
            "question": cleaned_question,
            "context": retrieval_result["fused_context"],
        }
    )

    return {
        "answer": response.content,
        "route": retrieval_result["route"],
        "query_intent": retrieval_result["query_intent"],
        "rewritten_query": retrieval_result["rewritten_query"],
        "local_documents": retrieval_result["local_documents"],
        "web_results": retrieval_result["web_results"],
        "fused_context": retrieval_result["fused_context"],
    }


# ============================================================
# Traditional local-only RAG chain
# ============================================================

@lru_cache(maxsize=1)
def get_retriever_runnable() -> RunnableLambda:
    """
    Return the cached LangChain runnable used for hybrid retrieval.
    """

    return RunnableLambda(run_hybrid_retrieval)


@lru_cache(maxsize=1)
def get_rag_prompt() -> ChatPromptTemplate:
    """
    Create and return the local-document RAG prompt.
    """

    system_message = """
You are PharmacologyGPT, a pharmacology question-answering assistant.

Answer the user's question using only the provided context.

Rules:
1. Do not use information outside the provided context.
2. If the answer is not found in the context, say:
   "I could not find the answer in the provided documents."
3. Give a clear and concise answer.
4. Do not invent drug facts, dosages, warnings, or medical advice.

Context:
{context}
"""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message,
            ),
            (
                "human",
                "{input}",
            ),
        ]
    )


@lru_cache(maxsize=1)
def get_document_chain():
    """
    Create and return the cached document-answering chain.
    """

    return create_stuff_documents_chain(
        llm=get_llm(),
        prompt=get_rag_prompt(),
    )


@lru_cache(maxsize=1)
def get_rag_chain():
    """
    Create and return the complete cached local RAG chain.
    """

    return create_retrieval_chain(
        get_retriever_runnable(),
        get_document_chain(),
    )