import os

from dotenv import load_dotenv
from tavily import TavilyClient


# Load environment variables from the local .env file.
load_dotenv()


def get_tavily_client() -> TavilyClient:
    """
    Create and return a Tavily API client.

    Raises:
        ValueError: If TAVILY_API_KEY is missing.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not tavily_api_key:
        raise ValueError(
            "TAVILY_API_KEY was not found. "
            "Add it to your .env file before running web search."
        )

    return TavilyClient(api_key=tavily_api_key)


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using Tavily and return structured results.

    Args:
        query: User search query.
        max_results: Maximum number of search results.

    Returns:
        A list of dictionaries containing title, URL, content, and score.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Search query cannot be empty.")

    if max_results < 1:
        raise ValueError("max_results must be at least 1.")

    tavily_client = get_tavily_client()

    response = tavily_client.search(
        query=cleaned_query,
        search_depth="basic",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )

    structured_results = []

    for result in response.get("results", []):
        structured_results.append(
            {
                "title": result.get("title", "Untitled result"),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score"),
            }
        )

    return structured_results