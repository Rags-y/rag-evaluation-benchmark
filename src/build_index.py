import argparse
import json
from pathlib import Path

import yaml

from rag_eval.vector_store import VectorStore


ROOT_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Build the vector index for the RAG benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--input-dir",
        default="data/processed",
        help="Directory containing processed corpus files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/index",
        help="Directory where the vector index will be saved.",
    )
    args = parser.parse_args()

    config = load_config(ROOT_DIR / args.config)

    input_dir = ROOT_DIR / args.input_dir
    chunks_path = input_dir / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing chunks file: {chunks_path}. "
            "Run build_corpus.py first."
        )

    chunks = json.loads(
        chunks_path.read_text(encoding="utf-8")
    )

    retrieval_config = config["retrieval"]
    embedding_model = retrieval_config["embedding_model"]

    print(f"Embedding model: {embedding_model}")
    print(f"Chunks: {len(chunks)}")

    vector_store = VectorStore(
        model_name=embedding_model,
    )

    vector_store.build(chunks)

    output_dir = ROOT_DIR / args.output_dir

    vector_store.save(output_dir)

    print()
    print("Vector index built successfully.")
    print(f"Index saved to: {output_dir}")


if __name__ == "__main__":
    main()