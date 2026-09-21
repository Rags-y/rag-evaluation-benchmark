from pathlib import Path

from rag_eval.vector_store import VectorStore


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_saved_index_can_be_loaded():
    index_dir = ROOT_DIR / "data" / "index"

    store = VectorStore()
    store.load(index_dir)

    assert store.embeddings is not None
    assert len(store.chunks) > 0


def test_python_list_query_returns_relevant_chunk():
    index_dir = ROOT_DIR / "data" / "index"

    store = VectorStore()
    store.load(index_dir)

    results = store.search(
        "How do Python lists work?",
        top_k=5,
    )

    assert len(results) == 5

    result_text = " ".join(
        result["text"].lower()
        for result in results
    )

    assert "list" in result_text


def test_retrieval_results_have_traceability():
    index_dir = ROOT_DIR / "data" / "index"

    store = VectorStore()
    store.load(index_dir)

    results = store.search(
        "What is a Python class?",
        top_k=3,
    )

    for result in results:
        assert "chunk_id" in result
        assert "document_id" in result
        assert "text" in result
        assert "score" in result
        assert "source_url" in result["metadata"]