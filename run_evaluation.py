import argparse
import json
from pathlib import Path

import yaml

from rag_eval.evaluation_dataset import EvaluationDataset
from rag_eval.evaluation_runner import EvaluationRunner
from rag_eval.evaluation_summary import summarize_results
from rag_eval.rag_pipeline import RAGPipeline


ROOT_DIR = Path(__file__).resolve().parent


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration."""

    with Path(config_path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Run the RAG evaluation benchmark."
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration.",
    )

    args = parser.parse_args()

    config = load_config(
        ROOT_DIR / args.config
    )

    retrieval_config = config["retrieval"]

    pipeline = RAGPipeline(
        index_dir=ROOT_DIR / "data" / "index",
        top_k=retrieval_config["top_k"],
        embedding_model=retrieval_config["embedding_model"],
    )

    dataset = EvaluationDataset(
        ROOT_DIR
        / "data"
        / "evaluation"
        / "questions.json"
    )

    runner = EvaluationRunner(
        pipeline=pipeline,
        dataset=dataset,
    )

    results = runner.run()

    summary = summarize_results(results)

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        results_dir / "baseline_results.json"
    )

    summary_path = (
        results_dir / "baseline_summary.json"
    )

    runner.save_results(
        results,
        results_path,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("RAG EVALUATION COMPLETE")
    print("=" * 80)

    print(f"Questions evaluated: {len(results)}")
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")

    print()
    print("Summary:")

    for question_type, metrics in summary.items():
        print()
        print(question_type)

        print(
            f"  Count: {metrics['count']}"
        )

        print(
            f"  Retrieval precision: "
            f"{metrics['retrieval_precision']:.3f}"
        )

        print(
            f"  Retrieval recall: "
            f"{metrics['retrieval_document_recall']:.3f}"
        )

        print(
            f"  Exact match: "
            f"{metrics['exact_match']:.3f}"
        )

        print(
            f"  Abstention rate: "
            f"{metrics['abstention_rate']:.3f}"
        )


if __name__ == "__main__":
    main()