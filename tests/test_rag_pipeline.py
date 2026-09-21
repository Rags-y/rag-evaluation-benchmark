from pathlib import Path

from rag_eval.rag_pipeline import RAGPipeline


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_rag_pipeline_returns_answer_and_context():
    pipeline = RAGPipeline(
        index_dir=ROOT_DIR / "data" / "index",
        top_k=3,
        embedding_model="all-MiniLM-L6-v2",
    )

    result = pipeline.answer(
        "What is a Python list?"
    )

    assert result["question"] == "What is a Python list?"
    assert result["answer"]
    assert len(result["contexts"]) == 3

    for context in result["contexts"]:
        assert "chunk_id" in context
        assert "document_id" in context
        assert "score" in context