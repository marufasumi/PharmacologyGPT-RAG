from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)


# Load environment variables from the .env file.
load_dotenv()


# Reopen the existing Chroma vector database.
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma(
    collection_name="pharmacology_books",
    persist_directory="./vector",
    embedding_function=embedding_model
)


# Create the MMR retriever.
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7,
    }
)


# Create the language model.
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0
)


# Create the RAG prompt.
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


# Create the document answering chain.
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=rag_prompt
)


# Connect the retriever and document chain.
rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)