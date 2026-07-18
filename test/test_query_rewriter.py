from query_rewriter import rewrite_web_query


def main():
    test_questions = [
        "Compare metformin pharmacology with recent safety evidence.",
        "What is the latest FDA warning for semaglutide?",
        "Tell me about current GLP-1 receptor agonist warnings.",
        "Was Ozempic approved for a new indication in 2026?",
    ]

    for question in test_questions:
        rewritten_query = rewrite_web_query(question)

        print("=" * 100)
        print(f"Original question: {question}")
        print(f"Rewritten query: {rewritten_query}")


if __name__ == "__main__":
    main()