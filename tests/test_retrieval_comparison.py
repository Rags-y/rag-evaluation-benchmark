from rag_eval.retrieval_comparison import compare_summaries


def make_summary(
    precision: float,
    recall: float,
    relevance: float,
    faithfulness: float,
    correctness: float,
    exact_match: float,
    abstention_rate: float,
):
    return {
        "direct_factual": {
            "count": 15,
            "retrieval_precision": precision,
            "retrieval_document_recall": recall,
            "answer_relevance": relevance,
            "faithfulness": faithfulness,
            "correctness": correctness,
            "exact_match": exact_match,
            "abstention_rate": abstention_rate,
        }
    }


def test_compare_summaries():
    baseline = make_summary(
        precision=0.68,
        recall=1.0,
        relevance=0.538,
        faithfulness=0.956,
        correctness=0.697,
        exact_match=0.0,
        abstention_rate=0.067,
    )

    variant = make_summary(
        precision=0.733,
        recall=0.933,
        relevance=0.494,
        faithfulness=1.0,
        correctness=0.607,
        exact_match=0.0,
        abstention_rate=0.0,
    )

    comparison = compare_summaries(
        baseline,
        variant,
    )

    metrics = comparison["direct_factual"]

    assert metrics["retrieval_precision"]["baseline"] == 0.68
    assert metrics["retrieval_precision"]["variant"] == 0.733

    assert round(
        metrics["retrieval_precision"]["delta"],
        3,
    ) == 0.053

    assert round(
        metrics["retrieval_document_recall"]["delta"],
        3,
    ) == -0.067

    assert round(
        metrics["abstention_rate"]["delta"],
        3,
    ) == -0.067