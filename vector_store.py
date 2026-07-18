"""
Shared Chroma Vector Store

This module lazily initializes and provides shared instances of:

1. The OpenAI embedding model
2. The persistent Chroma vector store

If the vector database is missing, the application downloads and
extracts the prebuilt Chroma database before initializing Chroma.

The objects are created only when requested and are then cached
for reuse throughout the application.
"""

from functools import lru_cache

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from vector_db_manager import download_vector_database


# Load environment variables from .env when running locally.
load_dotenv()

COLLECTION_NAME = "pharmacology_books"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    """
    Create and return the shared OpenAI embedding model.

    The embedding model is initialized only on the first call.
    Later calls return the same cached instance.
    """

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Create and return the shared persistent Chroma vector store.

    If the vector database is not available locally, it is downloaded
    and extracted first.

    The Chroma object is initialized only on the first call.
    Later calls return the same cached instance.
    """

    vector_directory = download_vector_database()

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(vector_directory),
        embedding_function=get_embedding_model(),
    )