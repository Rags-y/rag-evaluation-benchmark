from typing import Any


def retrieval_document_recall(
    retrieved_contexts: list[dict[str, Any]],
    expected_document_ids: list[str],
) -> float:
    """Measure whether expected source documents were retrieved."""

    if not expected_document_ids:
        return 0.0

    retrieved_document_ids = {
        context["document_id"]
        for context in retrieved_contexts
    }

    expected_ids = set(expected_document_ids)

    matched = retrieved_document_ids.intersection(
        expected_ids
    )

    return len(matched) / len(expected_ids)


def retrieval_precision(
    retrieved_contexts: list[dict[str, Any]],
    expected_document_ids: list[str],
) -> float:
    """Measure the proportion of retrieved contexts from expected documents."""

    if not retrieved_contexts:
        return 0.0

    if not expected_document_ids:
     return 0.0

    expected_ids = set(expected_document_ids)

    relevant = sum(
        1
        for context in retrieved_contexts
        if context["document_id"] in expected_ids
    )

    return relevant / len(retrieved_contexts)


def answer_exact_match(
    answer: str,
    expected_answer: str | None,
) -> float:
    """Simple normalized exact-match metric."""

    if expected_answer is None:
        return 0.0

    normalized_answer = answer.strip().lower()
    normalized_expected = expected_answer.strip().lower()

    return float(
        normalized_answer == normalized_expected
    )


def is_abstention(
    answer: str,
) -> bool:
    """Detect whether the system explicitly abstained."""

    abstention_phrases = [
        "i don't know",
        "i do not know",
        "i don't have enough information",
        "i do not have enough information",
        "cannot answer",
        "can't answer",
        "not enough information",
        "information is not available",
    ]

    normalized_answer = answer.strip().lower()

    return any(
        phrase in normalized_answer
        for phrase in abstention_phrases
    )