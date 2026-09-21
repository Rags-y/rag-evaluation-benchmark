from rag_eval.vector_store import VectorStore


def test_vector_store_search():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "text": "Python lists are mutable sequences.",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-2",
            "text": "Python classes provide a means of bundling data and functionality.",
        },
        {
            "chunk_id": "chunk-3",
            "document_id": "doc-3",
            "text": "The try statement allows you to test a block of code for errors.",
        },
    ]

    store = VectorStore()

    store.build(chunks)

    results = store.search(
        "What are Python lists?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-1"
    assert "score" in results[0]