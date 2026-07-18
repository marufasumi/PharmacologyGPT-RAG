from rag import retrieve_routed_context


def print_result(question: str) -> None:
    result = retrieve_routed_context(question)

    print("=" * 100)
    print(f"Question: {question}")
    print(f"Route: {result['route']}")
    print(f"Local documents: {len(result['local_documents'])}")
    print(f"Web results: {len(result['web_results'])}")
    print("\nFused context preview:\n")
    print(result["fused_context"][:1500])
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