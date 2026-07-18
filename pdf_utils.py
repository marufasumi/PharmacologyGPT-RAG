"""
PDF Processing Utilities

This module contains reusable functions for:

1. Loading PDF documents
2. Splitting documents into smaller chunks
3. Checking whether a PDF is already indexed
4. Adding new PDF chunks to the existing ChromaDB database
5. Returning vector database statistics
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import get_vectorstore


def load_and_split_pdf(pdf_path: str | Path) -> list:
    """
    Load a PDF and split its pages into smaller text chunks.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.

    Returns
    -------
    list
        A list of LangChain Document chunks.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file was not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"The supplied file is not a PDF: {pdf_path}"
        )

    # Load each PDF page as a LangChain Document.
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # Split large page content into smaller overlapping chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    return chunks


def is_pdf_indexed(pdf_path: str | Path) -> bool:
    """
    Check whether a PDF already exists in the vector database.

    This prevents the same PDF from being embedded and stored
    multiple times.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.

    Returns
    -------
    bool
        True when the PDF already has stored chunks.
        False when the PDF is not yet indexed.
    """

    # Initialize Chroma only when this function is called.
    vectorstore = get_vectorstore()

    # Convert the path into the string format used
    # inside the document metadata.
    source_path = str(Path(pdf_path))

    # Search ChromaDB for documents with this source path.
    stored_documents = vectorstore.get(
        where={
            "source": source_path,
        }
    )

    return len(stored_documents.get("ids", [])) > 0


def add_pdf_to_vectorstore(
    pdf_path: str | Path,
    batch_size: int = 250,
) -> int:
    """
    Load, split, embed, and store a PDF in ChromaDB.

    The chunks are stored in batches to make large-document
    processing more manageable.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.

    batch_size : int, optional
        Number of chunks added during each database operation.

    Returns
    -------
    int
        Total number of chunks added to ChromaDB.
        Returns 0 when the PDF is already indexed.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    pdf_path = Path(pdf_path)

    # Stop if this PDF has already been indexed.
    if is_pdf_indexed(pdf_path):
        return 0

    # Load and split the PDF.
    chunks = load_and_split_pdf(pdf_path)

    total_chunks = len(chunks)

    if total_chunks == 0:
        return 0

    # Initialize Chroma only when documents need to be stored.
    vectorstore = get_vectorstore()

    # Add chunks to ChromaDB in smaller batches.
    for start_index in range(
        0,
        total_chunks,
        batch_size,
    ):
        end_index = start_index + batch_size

        current_batch = chunks[
            start_index:end_index
        ]

        vectorstore.add_documents(current_batch)

        stored_count = min(
            end_index,
            total_chunks,
        )

        print(
            f"Stored {stored_count} "
            f"of {total_chunks} chunks"
        )

    return total_chunks


def get_vectorstore_statistics() -> tuple[int, int]:
    """
    Return live statistics from the Chroma vector database.

    The function calculates:

    1. Total number of stored chunks
    2. Total number of unique PDF source files

    Returns
    -------
    tuple[int, int]
        A tuple containing:

        - total_pdf_count
        - total_chunk_count
    """

    # Initialize Chroma only when statistics are requested.
    vectorstore = get_vectorstore()

    # Retrieve stored IDs and metadata from ChromaDB.
    stored_data = vectorstore.get(
        include=["metadatas"]
    )

    # Every stored ID represents one text chunk.
    total_chunk_count = len(
        stored_data.get("ids", [])
    )

    # Collect unique PDF source filenames.
    unique_pdf_sources = set()

    for metadata in stored_data.get(
        "metadatas",
        [],
    ):
        if not metadata:
            continue

        source = metadata.get("source")

        if source:
            source_name = Path(source).name
            unique_pdf_sources.add(source_name)

    total_pdf_count = len(unique_pdf_sources)

    return total_pdf_count, total_chunk_count