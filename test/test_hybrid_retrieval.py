from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

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
# Create the vector retriever
# ------------------------------------------------------------
vector_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 30,
        "lambda_mult": 0.7,
    },
)


# ------------------------------------------------------------
# Load documents and create BM25
# ------------------------------------------------------------
print("=" * 80)
print("HYBRID RETRIEVER SETUP")
print("=" * 80)

documents = load_documents_from_chroma(
    vectorstore=vectorstore,
)

bm25_retriever = create_bm25_retriever(
    documents=documents,
    result_count=10,
)

print(f"Documents loaded from Chroma: {len(documents)}")
print("Vector retriever created successfully.")
print("BM25 retriever created successfully.")


# ------------------------------------------------------------
# Pharmacology evaluation questions
# ------------------------------------------------------------
test_queries = [
    "What is the mechanism of action of metformin?",
    "What drugs inhibit CYP3A4?",
    "How does succinylcholine produce neuromuscular blockade?",
    "Why do ACE inhibitors cause cough?",
    "What is the mechanism of beta-2 agonists?",
]


# ------------------------------------------------------------
# Display helper
# ------------------------------------------------------------
def print_result_summary(
    results: list[Document],
) -> None:
    """
    Print a concise summary of hybrid retrieval results.
    """

    for rank, document in enumerate(results, start=1):
        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page",
            "Unknown page",
        )

        retrieval_sources = document.metadata.get(
            "retrieval_sources",
            [],
        )

        fusion_score = document.metadata.get(
            "fusion_score",
            0.0,
        )

        content_preview = " ".join(
            document.page_content.split()
        )[:300]

        print(f"\nResult {rank}")
        print("-" * 80)
        print(f"Source: {source}")
        print(f"Page: {page}")
        print(
            "Retrieved by: "
            f"{', '.join(retrieval_sources)}"
        )
        print(f"Fusion score: {fusion_score:.8f}")
        print(f"Preview: {content_preview}")


# ------------------------------------------------------------
# Run every evaluation query
# ------------------------------------------------------------
for query_number, query in enumerate(
    test_queries,
    start=1,
):
    hybrid_results = retrieve_hybrid_documents(
        query=query,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        vector_weight=0.5,
        bm25_weight=0.5,
        final_result_count=5,
    )

    print("\n\n" + "=" * 80)
    print(f"QUERY {query_number}")
    print("=" * 80)
    print(query)

    print_result_summary(
        results=hybrid_results,
    )