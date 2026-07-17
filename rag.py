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