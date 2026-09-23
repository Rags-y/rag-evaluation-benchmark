from collections import defaultdict
from typing import Any


def summarize_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate aggregate evaluation metrics."""

    groups = defaultdict(list)

    for result in results:
        groups[result["question_type"]].append(result)

    summary = {}

    for question_type, items in groups.items():
        summary[question_type] = {
            "count": len(items),
            "retrieval_precision": sum(
                item["retrieval_precision"]
                for item in items
            ) / len(items),
            "retrieval_document_recall": sum(
                item["retrieval_document_recall"]
                for item in items
            ) / len(items),
            "answer_relevance": sum(
                item["answer_relevance"]
                for item in items
            ) / len(items),
            "faithfulness": sum(
                item["faithfulness"]
                for item in items
            ) / len(items),
            "correctness": sum(
                item["correctness"]
                for item in items
            ) / len(items),
            "exact_match": sum(
                item["exact_match"]
                for item in items
            ) / len(items),
            "abstention_rate": sum(
                item["abstained"]
                for item in items
            ) / len(items),
        }

    return summary