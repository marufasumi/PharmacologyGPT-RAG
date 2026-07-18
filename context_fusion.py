from langchain_core.documents import Document


def format_local_documents(
    local_documents: list[Document],
) -> str:
    """
    Convert retrieved local documents into a readable context block.
    """
    if not local_documents:
        return "No local knowledge was retrieved."

    formatted_documents = []

    for document_number, document in enumerate(
        local_documents,
        start=1,
    ):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_document = (
            f"[Local Source {document_number}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{document.page_content.strip()}"
        )

        formatted_documents.append(formatted_document)

    return "\n\n".join(formatted_documents)


def format_web_results(
    web_results: list[dict],
) -> str:
    """
    Convert Tavily search results into a readable context block.
    """
    if not web_results:
        return "No web knowledge was retrieved."

    formatted_results = []

    for result_number, result in enumerate(
        web_results,
        start=1,
    ):
        title = result.get("title", "Untitled result")
        url = result.get("url", "")
        content = result.get("content", "").strip()

        formatted_result = (
            f"[Web Source {result_number}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content:\n{content}"
        )

        formatted_results.append(formatted_result)

    return "\n\n".join(formatted_results)


def build_fused_context(
    local_documents: list[Document] | None = None,
    web_results: list[dict] | None = None,
) -> str:
    """
    Combine local retrieval results and web search results
    into one structured context string.
    """
    local_documents = local_documents or []
    web_results = web_results or []

    local_context = format_local_documents(local_documents)
    web_context = format_web_results(web_results)

    fused_context = (
        "LOCAL KNOWLEDGE\n"
        "===============\n"
        f"{local_context}\n\n"
        "WEB KNOWLEDGE\n"
        "=============\n"
        f"{web_context}"
    )

    return fused_context