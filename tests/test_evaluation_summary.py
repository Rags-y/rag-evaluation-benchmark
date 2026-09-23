from rag_eval.evaluation_summary import summarize_results


def test_summarize_results():
    results = [
        {
            "question_type": "direct_factual",
            "retrieval_precision": 1.0,
            "retrieval_document_recall": 1.0,
            "answer_relevance": 1.0,
            "faithfulness": 1.0,
            "correctness": 1.0,
            "exact_match": 1.0,
            "abstained": False,
        },
        {
            "question_type": "direct_factual",
            "retrieval_precision": 0.5,
            "retrieval_document_recall": 0.0,
            "answer_relevance": 0.5,
            "faithfulness": 0.5,
            "correctness": 0.0,
            "exact_match": 0.0,
            "abstained": True,
        },
    ]

    summary = summarize_results(results)

    assert summary["direct_factual"]["count"] == 2

    assert (
        summary["direct_factual"]["retrieval_precision"]
        == 0.75
    )

    assert (
        summary["direct_factual"]["retrieval_document_recall"]
        == 0.5
    )

    assert (
        summary["direct_factual"]["answer_relevance"]
        == 0.75
    )

    assert (
        summary["direct_factual"]["faithfulness"]
        == 0.75
    )

    assert (
        summary["direct_factual"]["correctness"]
        == 0.5
    )

    assert (
        summary["direct_factual"]["exact_match"]
        == 0.5
    )

    assert (
        summary["direct_factual"]["abstention_rate"]
        == 0.5
    )