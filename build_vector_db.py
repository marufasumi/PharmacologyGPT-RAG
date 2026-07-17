"""
Build the Chroma Vector Database

Purpose
-------
This script indexes every PDF inside the docs folder.

Workflow

docs/
    ↓
Load PDFs
    ↓
Split into chunks
    ↓
Generate embeddings
    ↓
Store in ChromaDB

Run this script only when creating or rebuilding
the knowledge base.
"""

from pathlib import Path

from pdf_utils import add_pdf_to_vectorstore


# ---------------------------------------------------------
# Location of all PDF files
# ---------------------------------------------------------

pdf_folder = Path("docs")

pdf_files = sorted(
    pdf_folder.glob("*.pdf")
)


# ---------------------------------------------------------
# Index every PDF
# ---------------------------------------------------------

print("=" * 60)
print("Building Chroma Vector Database")
print("=" * 60)

total_pdfs = 0

for pdf in pdf_files:

    print(f"\nProcessing: {pdf.name}")

    chunks_added = add_pdf_to_vectorstore(
        str(pdf)
    )

    if chunks_added == 0:

        print("Already indexed")

    else:

        print(
            f"Added {chunks_added:,} chunks"
        )

    total_pdfs += 1


print("\n" + "=" * 60)
print("Finished")
print("=" * 60)

print(f"PDFs processed : {total_pdfs}")