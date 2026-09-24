import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_results(path: str | Path) -> list[dict]:
    """Load evaluation results."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def failure_score(result: dict) -> float:
    """Calculate a failure score while handling unanswerable cases."""

    answerable = result["answerable"]
    abstained = result["abstained"]

    # Correctly abstaining on an unanswerable question
    # is not considered a failure.
    if not answerable and abstained:
        return 0.0

    # Failing to abstain on an unanswerable question
    # is a significant failure.
    if not answerable and not abstained:
        return (
            1
            + (1 - result["faithfulness"])
            + (1 - result["answer_relevance"])
        )

    # Answerable questions:
    # combine retrieval and answer-quality weaknesses.
    score = (
        (1 - result["retrieval_precision"])
        + (1 - result["retrieval_document_recall"])
        + (1 - result["answer_relevance"])
        + (1 - result["faithfulness"])
        + (1 - result["correctness"])
    )

    # Additional penalty when the system abstains
    # from an answerable question.
    if abstained:
        score += 1

    return score


def get_worst_failures(
    results: list[dict],
    limit: int = 10,
) -> list[dict]:
    """Return the questions with the highest failure scores."""

    ranked = []

    for result in results:
        score = failure_score(result)

        if score <= 0:
            continue

        item = result.copy()
        item["failure_score"] = score
        ranked.append(item)

    ranked.sort(
        key=lambda item: item["failure_score"],
        reverse=True,
    )

    return ranked[:limit]


def save_worst_failures(
    failures: list[dict],
    output_path: str | Path,
) -> None:
    """Save worst failures as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            failures,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    results_path = (
        ROOT_DIR
        / "results"
        / "baseline_results.json"
    )

    output_path = (
        ROOT_DIR
        / "results"
        / "worst_failures.json"
    )

    results = load_results(results_path)

    failures = get_worst_failures(
        results,
        limit=10,
    )

    save_worst_failures(
        failures,
        output_path,
    )

    print()
    print("=" * 80)
    print("TOP 10 WORST FAILURES")
    print("=" * 80)

    for index, result in enumerate(failures, start=1):
        print()
        print(
            f"{index}. {result['question_id']} "
            f"({result['question_type']})"
        )
        print(
            f"   Failure score: "
            f"{result['failure_score']:.3f}"
        )
        print(
            f"   Question: "
            f"{result['question']}"
        )
        print(
            f"   Answer: "
            f"{result['answer'][:300]}"
        )

    print()
    print(
        f"Worst failures saved to: {output_path}"
    )


if __name__ == "__main__":
    main()