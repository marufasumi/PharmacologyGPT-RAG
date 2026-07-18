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

from pdf_utils import (
    add_pdf_to_vectorstore,
    get_vectorstore_statistics,
    is_pdf_indexed,
)
from rag import answer_routed_question


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="PharmacologyGPT",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("💊 PharmacologyGPT")
st.sidebar.caption("Hybrid pharmacology knowledge assistant")
st.sidebar.markdown("---")


# ---------------------------------------------------------
# Knowledge-base information
# ---------------------------------------------------------

st.sidebar.subheader("Knowledge Base")

total_pdf_count, total_chunk_count = get_vectorstore_statistics()

metric_col_1, metric_col_2 = st.sidebar.columns(2)

with metric_col_1:
    st.metric("PDFs", f"{total_pdf_count:,}")

with metric_col_2:
    st.metric("Chunks", f"{total_chunk_count:,}")

st.sidebar.success("Knowledge base ready")
st.sidebar.markdown("---")


# ---------------------------------------------------------
# AI-model information
# ---------------------------------------------------------

with st.sidebar.expander("⚙️ AI Model Configuration"):
    st.write("**Embeddings**")
    st.caption("text-embedding-3-small")

    st.write("**Language Model**")
    st.caption("GPT-5 Nano")

    st.write("**Retrieval**")
    st.caption("Vector search + BM25 + RRF")


# ---------------------------------------------------------
# PDF upload section
# ---------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("Add a Document")
st.sidebar.caption(
    "Upload a PDF to extend the local pharmacology knowledge base."
)

if "index_success_message" in st.session_state:
    st.sidebar.success(st.session_state["index_success_message"])
    del st.session_state["index_success_message"]

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)

if uploaded_file is not None:
    os.makedirs("docs", exist_ok=True)

    file_path = os.path.join("docs", uploaded_file.name)

    st.sidebar.info(f"Selected: {uploaded_file.name}")

    pdf_already_indexed = is_pdf_indexed(file_path)

    if pdf_already_indexed:
        st.sidebar.warning("This PDF is already indexed.")
    else:
        index_button = st.sidebar.button(
            "Index PDF",
            type="primary",
            use_container_width=True,
        )

        if index_button:
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            with st.sidebar.spinner("Processing and indexing PDF..."):
                total_chunks_added = add_pdf_to_vectorstore(file_path)

            if total_chunks_added > 0:
                st.session_state["index_success_message"] = (
                    "PDF indexed successfully. "
                    f"{total_chunks_added:,} chunks added."
                )
                st.rerun()
            else:
                st.sidebar.warning("The PDF was already indexed.")


# ---------------------------------------------------------
# Conversation controls
# ---------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("Conversation")

if st.sidebar.button(
    "🗑️ Clear Conversation",
    use_container_width=True,
):
    st.session_state.messages = []
    st.rerun()


# ---------------------------------------------------------
# Main page header
# ---------------------------------------------------------

st.title("💊 PharmacologyGPT")
st.subheader("Hybrid RAG Assistant for Pharmacology")
st.caption(
    "Ask questions using pharmacology textbooks, live web search, "
    "or both—with transparent source citations."
)

feature_col_1, feature_col_2, feature_col_3, feature_col_4 = st.columns(4)

with feature_col_1:
    st.info("📚 Local Textbooks")

with feature_col_2:
    st.success("🌐 Live Web Search")

with feature_col_3:
    st.warning("🔀 Hybrid Retrieval")

with feature_col_4:
    st.info("🔎 Source Citations")

st.markdown("---")


# ---------------------------------------------------------
# Example questions
# ---------------------------------------------------------

if not st.session_state.messages:
    st.markdown("#### Try an Example")

    example_col_1, example_col_2, example_col_3 = st.columns(3)

    with example_col_1:
        st.caption("📚 Local knowledge")
        st.write("What is the mechanism of action of metformin?")

    with example_col_2:
        st.caption("🌐 Current information")
        st.write("What is the latest FDA warning for semaglutide?")

    with example_col_3:
        st.caption("🔀 Hybrid retrieval")
        st.write("Compare warfarin interactions with current guidance.")

    st.markdown("---")


# ---------------------------------------------------------
# Helper function for rendering sources
# ---------------------------------------------------------

def render_sources(response: dict) -> None:
    """
    Display retrieval route, rewritten query, local sources,
    and web sources for one assistant response.
    """

    route_labels = {
        "local": "📚 Local knowledge",
        "web": "🌐 Web knowledge",
        "both": "🔀 Local + web knowledge",
    }

    route = response.get("route", "unknown")

    st.caption(
        "Retrieval route: "
        f"{route_labels.get(route, route)}"
    )

    rewritten_query = response.get("rewritten_query")

    if rewritten_query:
        st.caption(f"🔎 Web search query: {rewritten_query}")

    unique_local_sources = set()

    for document in response.get("local_documents", []):
        source_name = Path(
            document.metadata.get("source", "Unknown source")
        ).name

        page_number = document.metadata.get("page")

        if page_number is not None:
            page_number += 1
        else:
            page_number = "Unknown"

        unique_local_sources.add((source_name, page_number))

    sorted_local_sources = sorted(
        unique_local_sources,
        key=lambda source: (
            source[0].lower(),
            str(source[1]),
        ),
    )

    if sorted_local_sources:
        with st.expander(
            f"📚 Local Sources ({len(sorted_local_sources)})"
        ):
            for source_name, page_number in sorted_local_sources:
                st.write(f"• {source_name} — Page {page_number}")

    web_results = response.get("web_results", [])

    if web_results:
        with st.expander(f"🌐 Web Sources ({len(web_results)})"):
            for web_result in web_results:
                title = web_result.get("title", "Untitled source")
                url = web_result.get("url", "")

                if url:
                    st.markdown(f"- [{title}]({url})")
                else:
                    st.write(f"• {title}")


# ---------------------------------------------------------
# Display conversation history
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            response_data = message.get("response_data")

            if response_data:
                render_sources(response_data)


# ---------------------------------------------------------
# Chat input and response generation
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a pharmacology question..."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching local documents and trusted web sources..."
        ):
            try:
                response = answer_routed_question(question)
            except Exception as error:
                st.error(
                    "The request could not be completed. Please try again."
                )
                st.exception(error)
                st.stop()

        answer = response.get(
            "answer",
            "No answer was generated.",
        )

        st.markdown(answer)
        render_sources(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "response_data": response,
        }
    )


# ---------------------------------------------------------
# Medical disclaimer
# ---------------------------------------------------------

st.markdown("---")
st.caption(
    "⚠️ PharmacologyGPT is an educational application and does not "
    "provide medical advice, diagnosis, or treatment recommendations. "
    "Verify drug-related decisions using official prescribing "
    "information and qualified healthcare professionals."
)