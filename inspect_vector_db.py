from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# Load variables from the local .env file.
load_dotenv()


# Use the same embedding model and Chroma configuration as rag.py.
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# Reopen the existing persistent Chroma collection.
vectorstore = Chroma(
    collection_name="pharmacology_books",
    persist_directory="./vector",
    embedding_function=embedding_model,
)


# Read the stored chunk text, metadata, and Chroma IDs.
stored_data = vectorstore.get(
    include=["documents", "metadatas"]
)


documents = stored_data.get("documents", [])
metadatas = stored_data.get("metadatas", [])
document_ids = stored_data.get("ids", [])


print("=" * 60)
print("CHROMA VECTOR DATABASE INSPECTION")
print("=" * 60)

print(f"Stored document IDs: {len(document_ids)}")
print(f"Stored chunk texts:  {len(documents)}")
print(f"Stored metadata:     {len(metadatas)}")


# Verify that Chroma returned usable content.
if not documents:
    print("\nNo document chunks were found.")
else:
    print("\nFirst stored chunk:")
    print("-" * 60)
    print(documents[0][:500])

    print("\nFirst chunk metadata:")
    print("-" * 60)
    print(metadatas[0])

    print("\nFirst Chroma ID:")
    print("-" * 60)
    print(document_ids[0])