"""
PharmacologyGPT Streamlit Application

Main responsibilities:

1. Display the application interface
2. Accept PDF uploads
3. Save uploaded PDFs
4. Index uploaded PDFs into ChromaDB
5. Accept pharmacology questions
6. Display answers and source citations
"""

import os
from pathlib import Path

import streamlit as st

# from rag import rag_chain
from rag import answer_routed_question
from pdf_utils import (
    add_pdf_to_vectorstore,
    get_vectorstore_statistics,
    is_pdf_indexed
)


# ---------------------------------------------------------
# Configure the Streamlit page
# ---------------------------------------------------------

st.set_page_config(
    page_title="PharmacologyGPT",
    page_icon="💊",
    layout="wide"
)


# ---------------------------------------------------------
# Sidebar title
# ---------------------------------------------------------

st.sidebar.title("💊 PharmacologyGPT")

st.sidebar.markdown("---")


# ---------------------------------------------------------
# Knowledge-base information
# ---------------------------------------------------------

st.sidebar.write("### Knowledge Base")

# Read the current PDF and chunk counts directly
# from the Chroma vector database.
total_pdf_count, total_chunk_count = (
    get_vectorstore_statistics()
)

st.sidebar.write(
    f"📚 PDFs Loaded: {total_pdf_count:,}"
)

st.sidebar.write(
    f"📄 Chunks: {total_chunk_count:,}"
)

st.sidebar.markdown("---")


# ---------------------------------------------------------
# AI-model information
# ---------------------------------------------------------

st.sidebar.write("### AI Models")

st.sidebar.write(
    "🧠 Embeddings: text-embedding-3-small"
)

st.sidebar.write(
    "🤖 LLM: GPT-5 Nano"
)

st.sidebar.markdown("---")

st.sidebar.success("✅ Local RAG Ready")


# ---------------------------------------------------------
# PDF upload section
# ---------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.write("### Add a Document")

# Display the most recent indexing result.
if "index_success_message" in st.session_state:

    st.sidebar.success(
        st.session_state[
            "index_success_message"
        ]
    )

    # Remove the message after displaying it once.
    del st.session_state[
        "index_success_message"
    ]

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# Only display the indexing controls when a PDF
# has been selected.
if uploaded_file is not None:

    # Create the docs folder when it does not exist.
    os.makedirs(
        "docs",
        exist_ok=True
    )

    # Build the local path for the uploaded file.
    file_path = os.path.join(
        "docs",
        uploaded_file.name
    )

    st.sidebar.info(
        f"Selected: {uploaded_file.name}"
    )

    # Check whether the selected PDF is already stored
    # inside the vector database.
    pdf_already_indexed = is_pdf_indexed(
        file_path
    )

    if pdf_already_indexed:

        st.sidebar.warning(
            "This PDF is already indexed."
        )

    else:

        # Indexing happens only after the user clicks
        # this button. This prevents accidental indexing
        # during normal Streamlit reruns.
        index_button = st.sidebar.button(
            "Index PDF",
            type="primary"
        )

        if index_button:

            # Save the uploaded PDF inside the docs folder.
            with open(file_path, "wb") as file:

                file.write(
                    uploaded_file.getbuffer()
                )

            # Display a spinner while the document is
            # being processed and embedded.
            with st.sidebar.spinner(
                "Processing and indexing PDF..."
            ):

                total_chunks_added = (
                    add_pdf_to_vectorstore(
                        file_path
                    )
                )

            if total_chunks_added > 0:
               # Store the message so it remains visible
               # after Streamlit reruns.
                st.session_state["index_success_message"] = (
                    f"PDF indexed successfully. "
                    f"{total_chunks_added:,} chunks added."
                )
                # Rerun so the dynamic sidebar statistics
                # immediately reflect the new PDF.
                st.rerun()

            else:

                st.sidebar.warning(
                    "The PDF was already indexed."
                )


# ---------------------------------------------------------
# Main page
# ---------------------------------------------------------

st.title("💊 PharmacologyGPT")

st.write(
    "Ask questions about pharmacology using "
    "your local PDF knowledge base."
)


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

question = st.text_input(
    "Ask a question:"
)


# ---------------------------------------------------------
# Ask button
# ---------------------------------------------------------

if st.button("Ask"):

    # Prevent empty questions from being submitted.
    if question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        # # Run the question through the RAG pipeline.
        # with st.spinner(
        #     "Searching the documents..."
        # ):

        #     response = rag_chain.invoke(
        #         {
        #             "input": question
        #         }
        #     )
        # Route the question, retrieve the appropriate sources,
        # and generate a grounded answer.
        with st.spinner(
        "Searching local documents and trusted web sources..."
       ):

         response = answer_routed_question(
          question
       ) 

        # -------------------------------------------------
        # Display the generated answer
        # -------------------------------------------------

        st.subheader("Answer")

        st.write(
            response["answer"]
        )


        # -------------------------------------------------
        # Display retrieval route
        # -------------------------------------------------

        route_labels = {
            "local": "📚 Local knowledge",
            "web": "🌐 Web knowledge",
            "both": "📚 Local + 🌐 Web knowledge",
        }

        st.caption(
            f"Retrieval route: "
            f"{route_labels.get(response['route'], response['route'])}"
        )


        # -------------------------------------------------
        # Collect unique local PDF citations
        # -------------------------------------------------

        unique_local_sources = set()

        for document in response["local_documents"]:

            # Display only the PDF filename rather than
            # the complete local file path.
            source_name = Path(
                document.metadata.get(
                    "source",
                    "Unknown source"
                )
            ).name

            # LangChain PDF page indexes normally begin at zero.
            page_number = document.metadata.get(
                "page"
            )

            if page_number is not None:

                page_number += 1

            else:

                page_number = "Unknown"

            unique_local_sources.add(
                (
                    source_name,
                    page_number
                )
            )


        # -------------------------------------------------
        # Sort local PDF citations
        # -------------------------------------------------

        sorted_local_sources = sorted(
            unique_local_sources,
            key=lambda source: (
                source[0].lower(),
                str(source[1])
            )
        )


        # -------------------------------------------------
        # Display local PDF sources
        # -------------------------------------------------

        if sorted_local_sources:

            with st.expander(
                f"📚 Local Sources "
                f"({len(sorted_local_sources)})"
            ):

                for source_name, page_number in sorted_local_sources:

                    st.write(
                        f"• {source_name} "
                        f"— Page {page_number}"
                    )


        # -------------------------------------------------
        # Display web sources
        # -------------------------------------------------

        if response["web_results"]:

            with st.expander(
                f"🌐 Web Sources "
                f"({len(response['web_results'])})"
            ):

                for web_result in response["web_results"]:

                    title = web_result.get(
                        "title",
                        "Untitled source"
                    )

                    url = web_result.get(
                        "url",
                        ""
                    )

                    if url:

                        st.markdown(
                            f"- [{title}]({url})"
                        )

                    else:

                        st.write(
                            f"• {title}"
                        )

        # # -------------------------------------------------
        # # Collect unique source citations
        # # -------------------------------------------------

        # unique_sources = set()

        # for document in response["context"]:

        #     # Extract only the PDF filename instead of
        #     # displaying the complete file path.
        #     source_name = Path(
        #         document.metadata.get(
        #             "source",
        #             "Unknown source"
        #         )
        #     ).name

        #     # LangChain page numbers begin at zero.
        #     # Add one so users see normal page numbering.
        #     page_number = document.metadata.get(
        #         "page"
        #     )

        #     if page_number is not None:

        #         page_number += 1

        #     else:

        #         page_number = "Unknown"

        #     # A set removes duplicate citations.
        #     unique_sources.add(
        #         (
        #             source_name,
        #             page_number
        #         )
        #     )


        # # -------------------------------------------------
        # # Sort source citations
        # # -------------------------------------------------

        # sorted_sources = sorted(
        #     unique_sources,
        #     key=lambda source: (
        #         source[0].lower(),
        #         str(source[1])
        #     )
        # )


        # # -------------------------------------------------
        # # Display sources in an expandable section
        # # -------------------------------------------------

        # with st.expander(
        #     f"📄 View Sources "
        #     f"({len(sorted_sources)})"
        # ):

        #     for source_name, page_number in sorted_sources:

        #         st.write(
        #             f"• {source_name} "
        #             f"— Page {page_number}"
        #         )