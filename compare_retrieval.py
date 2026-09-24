import argparse
from pathlib import Path

from rag_eval.retrieval_comparison import (
    compare_summaries,
    load_summary,
    save_comparison,
)


ROOT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Compare two RAG evaluation summaries."
    )

    parser.add_argument(
        "--baseline",
        default="results/baseline_summary.json",
        help="Path to baseline summary JSON.",
    )

    parser.add_argument(
        "--variant",
        default="results/variant_small_chunks_summary.json",
        help="Path to variant summary JSON.",
    )

    parser.add_argument(
        "--output",
        default="results/retrieval_comparison.json",
        help="Path for comparison JSON.",
    )

    args = parser.parse_args()

    baseline = load_summary(
        ROOT_DIR / args.baseline
    )

    variant = load_summary(
        ROOT_DIR / args.variant
    )

    comparison = compare_summaries(
        baseline,
        variant,
    )

    output_path = ROOT_DIR / args.output

    save_comparison(
        comparison,
        output_path,
    )

    print()
    print("=" * 80)
    print("RETRIEVAL COMPARISON")
    print("=" * 80)

    for question_type, metrics in comparison.items():
        print()
        print(question_type)

        for metric, values in metrics.items():
            print(
                f"  {metric}: "
                f"{values['baseline']:.3f} -> "
                f"{values['variant']:.3f} "
                f"(delta: {values['delta']:+.3f})"
            )

    print()
    print(f"Comparison saved to: {output_path}")


if __name__ == "__main__":
    main()