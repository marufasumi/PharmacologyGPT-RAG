from query_intent import QueryIntent, detect_query_intent


def main():
    test_cases = [
        (
            "What is the latest FDA warning for semaglutide?",
            QueryIntent.FDA_WARNING,
        ),
        (
            "What are the adverse effects of Ozempic?",
            QueryIntent.ADVERSE_EFFECT,
        ),
        (
            "What drugs interact with warfarin?",
            QueryIntent.DRUG_INTERACTION,
        ),
        (
            "What is the latest clinical trial for tirzepatide?",
            QueryIntent.CLINICAL_TRIAL,
        ),
        (
            "Compare metformin and semaglutide.",
            QueryIntent.COMPARISON,
        ),
        (
            "What is the mechanism of action of metformin?",
            QueryIntent.GENERAL,
        ),
    ]

    for question, expected_intent in test_cases:
        detected_intent = detect_query_intent(question)

        print("=" * 100)
        print(f"Question: {question}")
        print(f"Expected intent: {expected_intent.value}")
        print(f"Detected intent: {detected_intent.value}")

        assert detected_intent == expected_intent

    print("=" * 100)
    print("All query intent tests passed.")


if __name__ == "__main__":
    main()