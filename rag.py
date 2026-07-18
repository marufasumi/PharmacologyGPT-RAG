from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from hybrid_retriever import (
    create_bm25_retriever,
    load_documents_from_chroma,
    retrieve_hybrid_documents,
)
from context_fusion import build_fused_context
from router import route_question
from web_search import search_web
from query_rewriter import rewrite_web_query


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv()


# ------------------------------------------------------------
# Reopen the existing Chroma vector database
# ------------------------------------------------------------
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma(
    collection_name="pharmacology_books",
    persist_directory="./vector",
    embedding_function=embedding_model,
)


# ------------------------------------------------------------
# Create the Chroma vector retriever
# ------------------------------------------------------------
# Retrieve a larger candidate pool before hybrid rank fusion.
vector_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 30,
        "lambda_mult": 0.7,
    },
)


# ------------------------------------------------------------
# Reconstruct the stored documents for BM25
# ------------------------------------------------------------
documents = load_documents_from_chroma(
    vectorstore=vectorstore,
)


# ------------------------------------------------------------
# Create the BM25 keyword retriever
# ------------------------------------------------------------
bm25_retriever = create_bm25_retriever(
    documents=documents,
    result_count=10,
)


# ------------------------------------------------------------
# Create the hybrid retrieval function
# ------------------------------------------------------------
def run_hybrid_retrieval(chain_input: dict):
    """
    Extract the user's question from the retrieval-chain input
    and run hybrid vector plus BM25 retrieval.
    """

    query = chain_input.get("input", "")

    return retrieve_hybrid_documents(
        query=query,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        vector_weight=0.5,
        bm25_weight=0.5,
        final_result_count=5,
    )


def retrieve_routed_context(question: str) -> dict:
    """
    Route the user's question and retrieve the appropriate context.

    Possible routes:
    - local: Hybrid retrieval only.
    - web: Web search only.
    - both: Hybrid retrieval and web search.
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    route = route_question(cleaned_question)

    local_documents = []
    web_results = []
    rewritten_query = None

    # Retrieve documents from the local vector database.
    if route in {"local", "both"}:
        local_documents = retrieve_hybrid_documents(
            query=cleaned_question,
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            vector_weight=0.5,
            bm25_weight=0.5,
            final_result_count=5,
        )

    # Rewrite the query before performing a web search.
    if route in {"web", "both"}:
        rewritten_query = rewrite_web_query(cleaned_question)

        web_results = search_web(
            query=rewritten_query,
            max_results=5,
        )

    # Combine local and web context into a single prompt context.
    fused_context = build_fused_context(
        local_documents=local_documents,
        web_results=web_results,
    )

    return {
        "route": route,
        "rewritten_query": rewritten_query,
        "local_documents": local_documents,
        "web_results": web_results,
        "fused_context": fused_context,
    }
def answer_routed_question(question: str) -> dict:
    """
    Route a question, retrieve the correct context, and generate
    a grounded answer using GPT-5 Nano.
    """
    retrieval_result = retrieve_routed_context(question)

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
            ("human", "{question}"),
        ]
    )

    answer_chain = answer_prompt | llm

    response = answer_chain.invoke(
        {
            "question": question,
            "context": retrieval_result["fused_context"],
        }
    )
    return {
        "answer": response.content,
        "route": retrieval_result["route"],
        "rewritten_query": retrieval_result["rewritten_query"],
        "local_documents": retrieval_result["local_documents"],
        "web_results": retrieval_result["web_results"],
        "fused_context": retrieval_result["fused_context"],
  }
# Convert the Python retrieval function into a LangChain Runnable.
# create_retrieval_chain can use this object like a normal retriever.
retriever = RunnableLambda(run_hybrid_retrieval)


# ------------------------------------------------------------
# Create the language model
# ------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
)


# ------------------------------------------------------------
# Create the RAG prompt
# ------------------------------------------------------------
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

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", "{input}"),
    ]
)


# ------------------------------------------------------------
# Create the document-answering chain
# ------------------------------------------------------------
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=rag_prompt,
)


# ------------------------------------------------------------
# Connect the hybrid retriever and document chain
# ------------------------------------------------------------
rag_chain = create_retrieval_chain(
    retriever,
    document_chain,
)