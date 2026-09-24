import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_comparison(path: str | Path) -> dict:
    """Load retrieval comparison results."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Comparison file not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def plot_retrieval_metrics(
    comparison: dict,
    output_path: str | Path,
) -> None:
    """Create a retrieval precision/recall comparison plot."""

    question_types = list(comparison.keys())

    baseline_precision = [
        comparison[q]["retrieval_precision"]["baseline"]
        for q in question_types
    ]

    variant_precision = [
        comparison[q]["retrieval_precision"]["variant"]
        for q in question_types
    ]

    baseline_recall = [
        comparison[q]["retrieval_document_recall"]["baseline"]
        for q in question_types
    ]

    variant_recall = [
        comparison[q]["retrieval_document_recall"]["variant"]
        for q in question_types
    ]

    x = range(len(question_types))

    width = 0.2

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - 1.5 * width for i in x],
        baseline_precision,
        width,
        label="Baseline Precision",
    )

    plt.bar(
        [i - 0.5 * width for i in x],
        variant_precision,
        width,
        label="Small Chunks Precision",
    )

    plt.bar(
        [i + 0.5 * width for i in x],
        baseline_recall,
        width,
        label="Baseline Recall",
    )

    plt.bar(
        [i + 1.5 * width for i in x],
        variant_recall,
        width,
        label="Small Chunks Recall",
    )

    plt.xticks(
        list(x),
        question_types,
        rotation=15,
    )

    plt.ylabel("Score")

    plt.title(
        "Retrieval Precision and Recall by Question Type"
    )

    plt.ylim(0, 1.05)

    plt.legend()

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def main():
    comparison_path = (
        ROOT_DIR
        / "results"
        / "retrieval_comparison.json"
    )

    output_path = (
        ROOT_DIR
        / "plots"
        / "retrieval_precision_recall.png"
    )

    comparison = load_comparison(
        comparison_path
    )

    plot_retrieval_metrics(
        comparison,
        output_path,
    )

    print(
        f"Plot saved to: {output_path}"
    )


if __name__ == "__main__":
    main()