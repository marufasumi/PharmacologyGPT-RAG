from router import route_question


def main():
    test_questions = [
        "What is the mechanism of action of metformin?",
        "Why do ACE inhibitors cause cough?",
        "What is the latest FDA warning for semaglutide?",
        "Was Ozempic approved for a new indication in 2026?",
        "Compare metformin pharmacology with recent safety evidence.",
        "What are the current warnings for GLP-1 receptor agonists?",
    ]

    print("\nQuestion Router Test\n")

    for question in test_questions:
        route = route_question(question)

        print("=" * 80)
        print(f"Question: {question}")
        print(f"Route: {route}")
        print()


if __name__ == "__main__":
    main()