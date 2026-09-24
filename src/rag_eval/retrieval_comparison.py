import json
from pathlib import Path
from typing import Any


METRICS = [
    "retrieval_precision",
    "retrieval_document_recall",
    "answer_relevance",
    "faithfulness",
    "correctness",
    "exact_match",
    "abstention_rate",
]


def load_summary(path: str | Path) -> dict[str, Any]:
    """Load an evaluation summary JSON file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def compare_summaries(
    baseline: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, Any]:
    """Compare metrics between two evaluation summaries."""

    comparison = {}

    question_types = sorted(
        set(baseline).intersection(variant)
    )

    for question_type in question_types:
        comparison[question_type] = {}

        for metric in METRICS:
            baseline_value = baseline[question_type][metric]
            variant_value = variant[question_type][metric]

            comparison[question_type][metric] = {
                "baseline": baseline_value,
                "variant": variant_value,
                "delta": variant_value - baseline_value,
            }

    return comparison


def save_comparison(
    comparison: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save comparison results as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            comparison,
            indent=2,
        ),
        encoding="utf-8",
    )