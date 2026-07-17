import re
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# ------------------------------------------------------------
# BM25 text preprocessing
# ------------------------------------------------------------
def preprocess_text(text: str) -> list[str]:
    """
    Convert text into normalized tokens for BM25 retrieval.

    This function:
    1. Converts text to lowercase.
    2. Removes punctuation.
    3. Preserves letters, numbers, and internal hyphens.
    4. Removes common question and stop words.

    Examples preserved:
    - metformin
    - cyp3a4
    - 5-ht3
    - beta-2
    """

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    tokens = re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        text.lower(),
    )

    return [
        token
        for token in tokens
        if token not in stop_words
    ]


# ------------------------------------------------------------
# Create a stable document identity
# ------------------------------------------------------------
def create_document_key(document: Document) -> str:
    """
    Create a stable key for duplicate detection.

    Vector retrieval and BM25 retrieval may return separate
    Document objects representing the same stored chunk.

    We therefore create the identity from:
    - source
    - page number
    - page content

    This avoids depending on an internal Chroma document ID.
    """

    source = str(
        document.metadata.get(
            "source",
            "unknown-source",
        )
    )

    page = str(
        document.metadata.get(
            "page",
            "unknown-page",
        )
    )

    normalized_content = " ".join(
        document.page_content.split()
    )

    return f"{source}|{page}|{normalized_content}"


# ------------------------------------------------------------
# Load documents from Chroma
# ------------------------------------------------------------
def load_documents_from_chroma(
    vectorstore: Chroma,
) -> list[Document]:
    """
    Read stored chunks from Chroma and reconstruct
    LangChain Document objects for BM25 indexing.
    """

    stored_data = vectorstore.get(
        include=["documents", "metadatas"]
    )

    stored_texts = stored_data.get("documents", [])
    stored_metadatas = stored_data.get("metadatas", [])
    stored_ids = stored_data.get("ids", [])

    if not (
        len(stored_texts)
        == len(stored_metadatas)
        == len(stored_ids)
    ):
        raise ValueError(
            "Chroma returned mismatched document, "
            "metadata, and ID counts."
        )

    documents = []

    for document_id, text, metadata in zip(
        stored_ids,
        stored_texts,
        stored_metadatas,
    ):
        if not text or not text.strip():
            continue

        document_metadata = dict(metadata)

        # Preserve the ID for debugging and inspection.
        document_metadata["chroma_id"] = document_id

        documents.append(
            Document(
                page_content=text,
                metadata=document_metadata,
            )
        )

    if not documents:
        raise ValueError(
            "No usable documents were found in Chroma."
        )

    return documents


# ------------------------------------------------------------
# Create BM25 retriever
# ------------------------------------------------------------
def create_bm25_retriever(
    documents: list[Document],
    result_count: int = 10,
) -> BM25Retriever:
    """
    Build an in-memory BM25 keyword retriever.
    """

    bm25_retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=preprocess_text,
    )

    bm25_retriever.k = result_count

    return bm25_retriever


# ------------------------------------------------------------
# Reciprocal Rank Fusion
# ------------------------------------------------------------
def reciprocal_rank_fusion(
    vector_results: list[Document],
    bm25_results: list[Document],
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    rrf_constant: int = 60,
    final_result_count: int = 5,
) -> list[Document]:
    """
    Combine vector and BM25 rankings using weighted
    Reciprocal Rank Fusion.

    Formula:

        score = weight / (rrf_constant + rank)

    Documents appearing in both lists receive contributions
    from both retrievers and therefore tend to rank higher.
    """

    if vector_weight < 0 or bm25_weight < 0:
        raise ValueError(
            "Retriever weights cannot be negative."
        )

    if vector_weight + bm25_weight == 0:
        raise ValueError(
            "At least one retriever weight must be positive."
        )

    if rrf_constant <= 0:
        raise ValueError(
            "RRF constant must be greater than zero."
        )

    if final_result_count <= 0:
        raise ValueError(
            "Final result count must be greater than zero."
        )

    fusion_scores: dict[str, float] = defaultdict(float)
    documents_by_key: dict[str, Document] = {}

    ranked_result_groups = [
        (vector_results, vector_weight, "vector"),
        (bm25_results, bm25_weight, "bm25"),
    ]

    for results, weight, retriever_name in ranked_result_groups:
        for rank, document in enumerate(
            results,
            start=1,
        ):
            document_key = create_document_key(document)

            fusion_scores[document_key] += (
                weight / (rrf_constant + rank)
            )

            # Keep the first available version of the document.
            if document_key not in documents_by_key:
                documents_by_key[document_key] = document

            # Add retrieval information for later debugging.
            existing_document = documents_by_key[document_key]

            retrieval_sources = existing_document.metadata.get(
                "retrieval_sources",
                [],
            )

            if retriever_name not in retrieval_sources:
                retrieval_sources = [
                    *retrieval_sources,
                    retriever_name,
                ]

            existing_document.metadata[
                "retrieval_sources"
            ] = retrieval_sources

    sorted_document_keys = sorted(
        fusion_scores,
        key=fusion_scores.get,
        reverse=True,
    )

    fused_documents = []

    for document_key in sorted_document_keys[
        :final_result_count
    ]:
        document = documents_by_key[document_key]

        document.metadata["fusion_score"] = (
            fusion_scores[document_key]
        )

        fused_documents.append(document)

    return fused_documents


# ------------------------------------------------------------
# Hybrid retrieval function
# ------------------------------------------------------------
def retrieve_hybrid_documents(
    query: str,
    vector_retriever,
    bm25_retriever: BM25Retriever,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    final_result_count: int = 5,
) -> list[Document]:
    """
    Run vector retrieval and BM25 retrieval, then combine
    their rankings using Reciprocal Rank Fusion.
    """

    if not query or not query.strip():
        raise ValueError(
            "The retrieval query cannot be empty."
        )

    vector_results = vector_retriever.invoke(query)
    bm25_results = bm25_retriever.invoke(query)

    return reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        final_result_count=final_result_count,
    )