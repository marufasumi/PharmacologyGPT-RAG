from rag import answer_routed_question


def print_result(question: str) -> None:
    result = answer_routed_question(question)

    print("=" * 100)
    print(f"Question: {question}")
    print(f"Route: {result['route']}")
    print(f"Local documents: {len(result['local_documents'])}")
    print(f"Web results: {len(result['web_results'])}")

    print("\nAnswer:\n")
    print(result["answer"])
    print()


def main():
    test_questions = [
        "What is the mechanism of action of metformin?",
        "What is the latest FDA warning for semaglutide?",
        "Compare metformin pharmacology with recent safety evidence.",
    ]

    for question in test_questions:
        print_result(question)


if __name__ == "__main__":
    main()