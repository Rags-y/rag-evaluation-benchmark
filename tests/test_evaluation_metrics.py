from rag_eval.evaluation_metrics import (
    answer_correctness,
    answer_exact_match,
    answer_faithfulness,
    answer_relevance,
    is_abstention,
    retrieval_document_recall,
    retrieval_precision,
)


def test_retrieval_document_recall():
    contexts = [
        {"document_id": "doc-1"},
        {"document_id": "doc-2"},
        {"document_id": "doc-1"},
    ]

    score = retrieval_document_recall(
        contexts,
        ["doc-1", "doc-2"],
    )

    assert score == 1.0


def test_retrieval_document_recall_partial():
    contexts = [
        {"document_id": "doc-1"},
    ]

    score = retrieval_document_recall(
        contexts,
        ["doc-1", "doc-2"],
    )

    assert score == 0.5


def test_unanswerable_question_has_zero_recall():
    contexts = [
        {"document_id": "doc-1"},
        {"document_id": "doc-2"},
    ]

    score = retrieval_document_recall(
        contexts,
        [],
    )

    assert score == 0.0


def test_retrieval_precision():
    contexts = [
        {"document_id": "doc-1"},
        {"document_id": "doc-2"},
        {"document_id": "doc-3"},
    ]

    score = retrieval_precision(
        contexts,
        ["doc-1", "doc-2"],
    )

    assert score == 2 / 3


def test_answer_exact_match():
    score = answer_exact_match(
        "Python lists are mutable.",
        "Python lists are mutable.",
    )

    assert score == 1.0


def test_answer_exact_match_is_case_insensitive():
    score = answer_exact_match(
        "PYTHON LISTS ARE MUTABLE.",
        "python lists are mutable.",
    )

    assert score == 1.0


def test_abstention_detection():
    assert is_abstention(
        "I don't have enough information to answer this question."
    )


def test_non_abstention():
    assert not is_abstention(
        "Python lists are mutable sequences."
    )


def test_answer_relevance():
    score = answer_relevance(
        "What are Python lists?",
        "Python lists are mutable sequences.",
    )

    assert score > 0.0


def test_answer_relevance_no_overlap():
    score = answer_relevance(
        "What are Python lists?",
        "The weather is sunny today.",
    )

    assert score == 0.0


def test_answer_faithfulness():
    contexts = [
        {
            "text": "Python lists are mutable sequences.",
        }
    ]

    score = answer_faithfulness(
        "Python lists are mutable sequences.",
        contexts,
    )

    assert score == 1.0


def test_answer_faithfulness_without_context():
    score = answer_faithfulness(
        "Python lists are mutable sequences.",
        [],
    )

    assert score == 0.0


def test_answer_correctness():
    score = answer_correctness(
        "Python lists are mutable sequences.",
        "Python lists are mutable sequences.",
    )

    assert score == 1.0


def test_answer_correctness_partial():
    score = answer_correctness(
        "Python lists are mutable.",
        "Python lists are mutable sequences.",
    )

    assert 0.0 < score < 1.0


def test_answer_correctness_unanswerable():
    score = answer_correctness(
        "Paris is the capital of France.",
        None,
    )

    assert score == 0.0