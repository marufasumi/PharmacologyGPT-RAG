# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


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

Your goal is to improve retrieval quality for current medical and drug-safety information.

Rules:
1. Return only the rewritten search query.
2. Do not answer the question.
3. Remove conversational wording such as:
   - compare
   - explain
   - tell me
   - what is
   - can you
4. Preserve drug names, indications, adverse effects, agencies, and dates.
5. Add useful current-evidence terms when appropriate, such as:
   - latest
   - safety update
   - FDA
   - EMA
   - clinical evidence
   - adverse events
6. Keep the query concise.
7. Do not add facts that were not present in the original question.
""",
        ),
        (
            "human",
            "Original question: {question}"
        ),
    ]
)


query_rewriter_chain = (
    query_rewriter_prompt
    | query_rewriter_llm
)


def rewrite_web_query(question: str) -> str:
    """
    Rewrite a user question into a focused web search query.

    Args:
        question: Original user question.

    Returns:
        Rewritten search query.
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "Question cannot be empty."
        )

    response = query_rewriter_chain.invoke(
        {
            "question": cleaned_question
        }
    )

    rewritten_query = response.content.strip()

    if not rewritten_query:
        return cleaned_question

    return rewritten_query