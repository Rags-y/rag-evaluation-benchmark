import argparse
from pathlib import Path

import yaml

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
        description="Run the baseline RAG pipeline."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question to ask the RAG system.",
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

    result = pipeline.answer(
        args.query
    )

    print()
    print("=" * 80)
    print("BASELINE RAG")
    print("=" * 80)

    print()
    print(f"Question: {result['question']}")

    print()
    print("Answer:")
    print(result["answer"])

    print()
    print("Retrieved Contexts:")
    print("-" * 80)

    for rank, context in enumerate(
        result["contexts"],
        start=1,
    ):
        print()
        print(f"Rank: {rank}")
        print(f"Score: {context['score']:.4f}")
        print(f"Chunk ID: {context['chunk_id']}")
        print(f"Document: {context['document_id']}")
        print(f"Source: {context['metadata']['source_url']}")
        print(f"Text: {context['text'][:500]}...")


if __name__ == "__main__":
    main()