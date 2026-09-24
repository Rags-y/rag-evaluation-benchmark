import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]


def main():
    comparison_path = (
        ROOT_DIR
        / "results"
        / "retrieval_comparison.json"
    )

    output_path = (
        ROOT_DIR
        / "plots"
        / "answer_quality_comparison.png"
    )

    comparison = json.loads(
        comparison_path.read_text(encoding="utf-8")
    )

    question_types = list(comparison.keys())

    baseline_relevance = [
        comparison[q]["answer_relevance"]["baseline"]
        for q in question_types
    ]

    variant_relevance = [
        comparison[q]["answer_relevance"]["variant"]
        for q in question_types
    ]

    baseline_correctness = [
        comparison[q]["correctness"]["baseline"]
        for q in question_types
    ]

    variant_correctness = [
        comparison[q]["correctness"]["variant"]
        for q in question_types
    ]

    x = range(len(question_types))
    width = 0.2

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - 1.5 * width for i in x],
        baseline_relevance,
        width,
        label="Baseline Relevance",
    )

    plt.bar(
        [i - 0.5 * width for i in x],
        variant_relevance,
        width,
        label="Small Chunks Relevance",
    )

    plt.bar(
        [i + 0.5 * width for i in x],
        baseline_correctness,
        width,
        label="Baseline Correctness",
    )

    plt.bar(
        [i + 1.5 * width for i in x],
        variant_correctness,
        width,
        label="Small Chunks Correctness",
    )

    plt.xticks(
        list(x),
        question_types,
        rotation=15,
    )

    plt.ylabel("Score")

    plt.title(
        "Answer Relevance and Correctness by Question Type"
    )

    plt.ylim(0, 1.05)

    plt.legend()

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Plot saved to: {output_path}"
    )


if __name__ == "__main__":
    main()