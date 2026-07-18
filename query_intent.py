from enum import Enum


class QueryIntent(str, Enum):
    """
    Supported web search intents.
    """

    FDA_WARNING = "fda_warning"
    ADVERSE_EFFECT = "adverse_effect"
    DRUG_INTERACTION = "drug_interaction"
    CLINICAL_TRIAL = "clinical_trial"
    COMPARISON = "comparison"
    GENERAL = "general"


def detect_query_intent(question: str) -> QueryIntent:
    """
    Detect the user's search intent using simple keyword rules.

    Args:
        question: User question.

    Returns:
        QueryIntent
    """
    question = question.lower()

    # Highest priority: FDA and regulatory questions
    if any(keyword in question for keyword in [
        "warning",
        "recall",
        "fda",
        "boxed warning",
        "black box",
    ]):
        return QueryIntent.FDA_WARNING

    # Comparison should take priority over other content-specific intents
    if any(keyword in question for keyword in [
        "compare",
        "comparison",
        "versus",
        "vs",
    ]):
        return QueryIntent.COMPARISON

    # Drug interaction questions
    if any(keyword in question for keyword in [
        "interaction",
        "interact",
        "contraindication",
    ]):
        return QueryIntent.DRUG_INTERACTION

    # Clinical trial questions
    if any(keyword in question for keyword in [
        "trial",
        "study",
        "phase",
        "clinical",
    ]):
        return QueryIntent.CLINICAL_TRIAL

    # Adverse effect and safety questions
    if any(keyword in question for keyword in [
        "adverse",
        "side effect",
        "side effects",
        "toxicity",
        "safety",
    ]):
        return QueryIntent.ADVERSE_EFFECT

    # Default intent
    return QueryIntent.GENERAL