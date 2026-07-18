from query_intent import detect_query_intent
from query_rewriter import rewrite_web_query


def main():
    test_questions = [
        "What is the latest FDA warning for semaglutide?",
        "What are the adverse effects of Ozempic?",
        "What drugs interact with warfarin?",
        "What is the latest clinical trial for tirzepatide?",
        "Compare metformin and semaglutide.",
        "What is the mechanism of action of metformin?",
    ]

    for question in test_questions:
        
        rewritten_query, detected_intent = rewrite_web_query(question)

        print("=" * 100)
        print(f"Original question: {question}")
        print(f"Detected intent: {detected_intent.value}")
        print(f"Rewritten query: {rewritten_query}")


if __name__ == "__main__":
    main()