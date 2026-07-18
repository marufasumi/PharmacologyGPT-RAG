# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from query_intent import QueryIntent, detect_query_intent


# Load environment variables from the local .env file.
load_dotenv()


# Use a small, fast model because rewriting is a lightweight task.
query_rewriter_llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0
)


query_rewriter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You rewrite pharmacology questions into focused web search queries.

The detected intent is: {intent}

Intent-specific rules:

- fda_warning:
  Focus on the latest FDA warning, safety communication, recall, label update,
  boxed warning, or regulatory action.

- adverse_effect:
  Focus on recent adverse effects, toxicity, safety evidence,
  pharmacovigilance findings, or adverse-event data.

- drug_interaction:
  Focus on drug-drug interactions, contraindications,
  interaction mechanisms, and current clinical guidance.

- clinical_trial:
  Focus on the latest clinical trials, study results,
  trial phases, efficacy, and safety findings.

- comparison:
  Focus on recent clinical evidence comparing the named drugs,
  including efficacy, pharmacology, and safety when requested.

- general:
  Preserve the original pharmacology topic and create a concise,
  medically relevant search query.

General rules:
1. Return only the rewritten search query.
2. Do not answer the question.
3. Preserve drug names, agencies, indications, adverse effects, and dates.
4. Never add a year, date, drug name, indication, or agency that was not
   present in the original question.
5. Do not add brand-to-generic or generic-to-brand mappings unless both
   names were present in the original question.
6. Remove unnecessary conversational wording.
7. Keep the query concise and natural.
8. Avoid broad phrases such as "comprehensive list."
9. Do not invent facts.
""",
        ),
        (
            "human",
            "Original question: {question}",
        ),
    ]
)


query_rewriter_chain = (
    query_rewriter_prompt
    | query_rewriter_llm
)


def rewrite_web_query(question: str) -> tuple[str, QueryIntent]:
    """
    Rewrite a user question into an intent-aware web search query.

    Returns:
        Tuple containing:
        - rewritten search query
        - detected query intent
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    intent = detect_query_intent(cleaned_question)

    response = query_rewriter_chain.invoke(
        {
            "question": cleaned_question,
            "intent": intent.value,
        }
    )

    rewritten_query = response.content.strip()

    if not rewritten_query:
        rewritten_query = cleaned_question

    return rewritten_query, intent