from typing import Literal


# These are the only valid routing decisions.
RouteType = Literal["local", "web", "both"]


# Terms that usually indicate the user needs current information.
WEB_KEYWORDS = {
    "latest",
    "recent",
    "current",
    "today",
    "new",
    "updated",
    "update",
    "news",
    "warning",
    "recall",
    "approval",
    "approved",
    "fda",
    "ema",
    "2025",
    "2026",
}


# Terms suggesting the user wants both textbook knowledge
# and recent external information.
COMPARISON_KEYWORDS = {
    "compare",
    "comparison",
    "versus",
    "vs",
    "latest evidence",
    "current evidence",
    "recent evidence",
}


def route_question(question: str) -> RouteType:
    """
    Decide whether a question should use local retrieval,
    web search, or both.

    Args:
        question: The user's pharmacology question.

    Returns:
        One of:
        - "local"
        - "web"
        - "both"
    """
    cleaned_question = question.strip().lower()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    uses_comparison_language = any(
        keyword in cleaned_question
        for keyword in COMPARISON_KEYWORDS
    )

    uses_web_language = any(
        keyword in cleaned_question
        for keyword in WEB_KEYWORDS
    )

    if uses_comparison_language and uses_web_language:
        return "both"

    if uses_web_language:
        return "web"

    return "local"