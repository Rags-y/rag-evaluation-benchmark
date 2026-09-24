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
        / "abstention_rate_comparison.png"
    )

    comparison = json.loads(
        comparison_path.read_text(encoding="utf-8")
    )

    question_types = list(comparison.keys())

    baseline_rates = [
        comparison[q]["abstention_rate"]["baseline"]
        for q in question_types
    ]

    variant_rates = [
        comparison[q]["abstention_rate"]["variant"]
        for q in question_types
    ]

    x = range(len(question_types))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - width / 2 for i in x],
        baseline_rates,
        width,
        label="Baseline",
    )

    plt.bar(
        [i + width / 2 for i in x],
        variant_rates,
        width,
        label="Small Chunks",
    )

    plt.xticks(
        list(x),
        question_types,
        rotation=15,
    )

    plt.ylabel("Abstention Rate")

    plt.title(
        "Abstention Rate by Question Type"
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