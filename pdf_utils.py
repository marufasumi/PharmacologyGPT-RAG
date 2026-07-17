"""
PDF Processing Utilities

This module contains reusable functions for:

1. Loading PDF documents
2. Splitting documents into smaller chunks
3. Checking whether a PDF is already indexed
4. Adding new PDF chunks to the existing ChromaDB database
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import vectorstore


def load_and_split_pdf(pdf_path):
    """
    Load a PDF and split its pages into smaller text chunks.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    list
        A list of LangChain Document chunks.
    """

    # Load each PDF page as a LangChain Document.
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split large page content into smaller overlapping chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks


def is_pdf_indexed(pdf_path):
    """
    Check whether a PDF already exists in the vector database.

    This prevents the same PDF from being embedded and stored
    multiple times.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    bool
        True when the PDF already has stored chunks.
        False when the PDF is not yet indexed.
    """

    # Convert the path into the same string format used
    # inside the document metadata.
    source_path = str(Path(pdf_path))

    # Search ChromaDB for documents with this source path.
    stored_documents = vectorstore.get(
        where={
            "source": source_path
        }
    )

    return len(stored_documents["ids"]) > 0

def get_vectorstore_statistics():
    """
    Return live statistics from the Chroma vector database.

    The function calculates:

    1. Total number of stored chunks
    2. Total number of unique PDF source files

    Returns
    -------
    tuple
        A tuple containing:

        - total_pdf_count
        - total_chunk_count
    """

    # Retrieve stored IDs and metadata from ChromaDB.
    stored_data = vectorstore.get(
        include=["metadatas"]
    )

    # Every stored ID represents one text chunk.
    total_chunk_count = len(
        stored_data.get("ids", [])
    )

    # Collect unique source filenames.
    unique_pdf_sources = set()

    for metadata in stored_data.get(
        "metadatas",
        []
    ):

        if not metadata:
            continue

        source = metadata.get("source")

        if source:
            source_name = Path(source).name
            unique_pdf_sources.add(source_name)

    total_pdf_count = len(unique_pdf_sources)

    return total_pdf_count, total_chunk_count

def add_pdf_to_vectorstore(pdf_path, batch_size=250):
    """
    Load, split, embed, and store a PDF in ChromaDB.

    The chunks are stored in batches to make large-document
    processing more manageable.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    batch_size : int, optional
        Number of chunks added during each database operation.

    Returns
    -------
    int
        Total number of chunks added to ChromaDB.
    """

    # Stop if this PDF has already been indexed.
    if is_pdf_indexed(pdf_path):
        return 0

    # Load and split the PDF.
    chunks = load_and_split_pdf(pdf_path)

    total_chunks = len(chunks)

    # Add chunks to ChromaDB in smaller batches.
    for start_index in range(0, total_chunks, batch_size):

        end_index = start_index + batch_size

        current_batch = chunks[
            start_index:end_index
        ]

        vectorstore.add_documents(current_batch)

        stored_count = min(
            end_index,
            total_chunks
        )

        print(
            f"Stored {stored_count} "
            f"of {total_chunks} chunks"
        )

    return total_chunks


def get_vectorstore_statistics():
    """
    Return the number of unique indexed PDFs and total chunks
    currently stored in the Chroma vector database.

    Returns
    -------
    tuple
        total_pdf_count, total_chunk_count
    """

    # Retrieve chunk IDs and metadata from ChromaDB.
    stored_data = vectorstore.get(
        include=["metadatas"]
    )

    # Each stored ID represents one text chunk.
    total_chunk_count = len(
        stored_data.get("ids", [])
    )

    # Collect the unique source PDF filenames.
    unique_pdf_sources = set()

    for metadata in stored_data.get("metadatas", []):

        if not metadata:
            continue

        source = metadata.get("source")

        if source:
            source_name = Path(source).name
            unique_pdf_sources.add(source_name)

    total_pdf_count = len(unique_pdf_sources)

    return total_pdf_count, total_chunk_count