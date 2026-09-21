import json
from pathlib import Path

import yaml

from rag_eval.vector_store import VectorStore


ROOT_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration."""

    with Path(config_path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def main():
    config_path = ROOT_DIR / "configs" / "default.yaml"

    config = load_config(config_path)

    chunks_path = (
        ROOT_DIR
        / "data"
        / "processed"
        / "chunks.json"
    )

    index_dir = ROOT_DIR / "data" / "index"

    with chunks_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks)} chunks.")

    embedding_model = config["retrieval"]["embedding_model"]

    print(
        f"Embedding model: {embedding_model}"
    )

    store = VectorStore(
        model_name=embedding_model,
    )

    store.build(chunks)
    store.save(index_dir)

    print()
    print("Vector index built successfully.")
    print(f"Index directory: {index_dir}")
    print(
        f"Embedding shape: {store.embeddings.shape}"
    )


if __name__ == "__main__":
    main()