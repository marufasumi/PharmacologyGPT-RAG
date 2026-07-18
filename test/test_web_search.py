from web_search import search_web


def main():
    query = "latest FDA safety warning for semaglutide"

    print(f"\nQuery: {query}\n")

    results = search_web(
        query=query,
        max_results=5,
    )

    if not results:
        print("No search results were returned.")
        return

    for result_number, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {result_number}")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Score: {result['score']}")
        print(f"Content: {result['content']}")
        print()


if __name__ == "__main__":
    main()