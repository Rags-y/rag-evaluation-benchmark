from pathlib import Path

from rag_eval.vector_store import VectorStore


ROOT_DIR = Path(__file__).resolve().parent.parent


def main():
    index_dir = ROOT_DIR / "data" / "index"

    store = VectorStore()
    store.load(index_dir)

    query = "How do Python lists work?"

    results = store.search(
        query,
        top_k=3,
    )

    print()
    print(f"Query: {query}")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Document: {result['document_id']}")
        print(f"Source: {result['metadata']['source_url']}")
        print(f"Text: {result['text'][:500]}...")


if __name__ == "__main__":
    main()