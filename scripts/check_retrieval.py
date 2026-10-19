import os
from pathlib import Path

from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_ROOT / ".env")

from agent.vector_manager import VectorSearchManager  # noqa: E402


def main() -> None:
    namespace = os.getenv("RETRIEVAL_NAMESPACE")
    if not namespace:
        print("Set RETRIEVAL_NAMESPACE to the namespace you want to inspect.")
        return

    vector_manager = VectorSearchManager(namespace=namespace)
    print(f"Vector store stats: {vector_manager.get_stats()}")

    queries = ["what is rag", "rag", "document content"]
    for query in queries:
        print(f"\nQuery: {query}")
        results = vector_manager.search(query=query, k=5)

        if not results:
            print("No results found.")
            continue

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}: {result.page_content[:200]}")
            print(f"Metadata: {result.metadata}")


if __name__ == "__main__":
    main()
