
"""
==========================================================
Test PDF Indexing

Purpose:
This script tests whether a new PDF can be successfully
added to the existing Chroma vector database.

This is only for testing.

The main Streamlit application will eventually call
these functions automatically when a user uploads
a PDF.
==========================================================
"""


from pdf_utils import add_pdf_to_vectorstore


pdf_path = "docs/Basic and Clinical Pharmacology.pdf"

total_chunks = add_pdf_to_vectorstore(pdf_path)

print(
    f"\nPDF indexing completed. "
    f"Total chunks added: {total_chunks}"
)





# from pathlib import Path

# from langchain_community.document_loaders import PyPDFLoader
# from pdf_utils import load_and_split_pdf


# # ---------------------------------------------------------
# # Show all available PDFs
# # ---------------------------------------------------------

# pdf_folder = Path("docs")

# pdf_files = list(pdf_folder.glob("*.pdf"))

# print("=" * 60)
# print("Available PDFs")
# print("=" * 60)

# for pdf in pdf_files:
#     print(pdf.name)


# # ---------------------------------------------------------
# # Select a PDF to test
# # ---------------------------------------------------------

# pdf_path = "docs/Basic and Clinical Pharmacology.pdf"


# # ---------------------------------------------------------
# # Test the load_and_split_pdf() function
# # ---------------------------------------------------------

# chunks = load_and_split_pdf(pdf_path)

# print("\n" + "=" * 60)
# print(f"Chunks created: {len(chunks)}")


# # ---------------------------------------------------------
# # Inspect the original PDF
# # ---------------------------------------------------------

# loader = PyPDFLoader(pdf_path)

# documents = loader.load()

# total_characters = sum(
#     len(document.page_content)
#     for document in documents
# )

# print(f"PDF pages: {len(documents)}")
# print(f"Extracted characters: {total_characters:,}")


# print("\n" + "=" * 60)
# print("First page preview")
# print("=" * 60)

# print(documents[0].page_content[:500])

# from pathlib import Path

# from rag import vectorstore


# target_file = "Basic and Clinical Pharmacology.pdf"

# results = vectorstore.get(
#     where={
#         "source": f"docs/{target_file}"
#     }
# )

# print(f"Stored chunks for {target_file}: {len(results['ids'])}")