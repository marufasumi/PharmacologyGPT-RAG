from dotenv import load_dotenv
import re

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv()


def preprocess_text(text: str) -> list[str]:
    """
    Convert text into normalized tokens for BM25.

    This preprocessing:
    1. Converts text to lowercase.
    2. Removes punctuation.
    3. Preserves medical terms containing letters,
       numbers, and internal hyphens.
    4. Removes common English stop words.
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
# Read all documents from Chroma
# ------------------------------------------------------------
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
        "Chroma returned mismatched document counts."
    )


# ------------------------------------------------------------
# Convert Chroma data into LangChain Documents
# ------------------------------------------------------------
documents = []

for document_id, text, metadata in zip(
    stored_ids,
    stored_texts,
    stored_metadatas,
):
    if not text.strip():
        continue

    metadata = dict(metadata)
    metadata["chroma_id"] = document_id

    documents.append(
        Document(
            page_content=text,
            metadata=metadata,
        )
    )


print("=" * 70)
print("BM25 INDEX SETUP")
print("=" * 70)
print(f"Documents loaded for BM25: {len(documents)}")


# ------------------------------------------------------------
# Build BM25 Retriever
# ------------------------------------------------------------
bm25_retriever = BM25Retriever.from_documents(
    documents,
    preprocess_func=preprocess_text,
)

bm25_retriever.k = 5


# ------------------------------------------------------------
# Build existing Chroma MMR Retriever
# ------------------------------------------------------------
vector_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7,
    },
)


# ------------------------------------------------------------
# Test Query
# ------------------------------------------------------------
test_query = "What is the mechanism of action of metformin?"

print("\n" + "=" * 70)
print("TEST QUERY")
print("=" * 70)
print(test_query)


# ------------------------------------------------------------
# Retrieve Results
# ------------------------------------------------------------
bm25_results = bm25_retriever.invoke(test_query)
vector_results = vector_retriever.invoke(test_query)


# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------
def print_results(
    title: str,
    results: list[Document],
) -> None:
    """Pretty-print retrieved documents."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for rank, document in enumerate(results, start=1):

        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page",
            "Unknown page",
        )

        chroma_id = document.metadata.get(
            "chroma_id",
            "Not available",
        )

        print(f"\nRESULT {rank}")
        print("-" * 70)
        print(f"Source:    {source}")
        print(f"Page:      {page}")
        print(f"Chroma ID: {chroma_id}")

        print("\nContent:")
        print(document.page_content[:700])


# ------------------------------------------------------------
# Display BM25 Results
# ------------------------------------------------------------
print_results(
    title="BM25 KEYWORD RESULTS",
    results=bm25_results,
)


# ------------------------------------------------------------
# Display Vector Results
# ------------------------------------------------------------
print_results(
    title="CHROMA VECTOR RESULTS",
    results=vector_results,
)