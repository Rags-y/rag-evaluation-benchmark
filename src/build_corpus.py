import argparse
import json
from pathlib import Path

import yaml

from rag_eval.chunker import TextChunker
from rag_eval.loader import DocumentLoader


ROOT_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Build the RAG benchmark corpus."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where processed corpus files will be saved.",
    )
    args = parser.parse_args()

    config = load_config(ROOT_DIR / args.config)

    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = ROOT_DIR / "data" / "raw"

    loader = DocumentLoader(raw_dir=raw_dir)

    documents = []

    for document_config in config["corpus"]["documents"]:
        document = loader.load(
            document_id=document_config["id"],
            title=document_config["title"],
            url=document_config["url"],
        )

        documents.append(
            {
                "document_id": document.document_id,
                "title": document.title,
                "source_url": document.source_url,
                "text": document.text,
                "metadata": document.metadata,
            }
        )

        print(
            f"Loaded: {document.title} "
            f"({len(document.text)} characters)"
        )

    chunking_config = config["chunking"]

    chunker = TextChunker(
        chunk_size=chunking_config["chunk_size"],
        chunk_overlap=chunking_config["chunk_overlap"],
    )

    chunks = []

    for document in documents:
        document_chunks = chunker.chunk_document(
            type(
                "Document",
                (),
                {
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "source_url": document["source_url"],
                    "text": document["text"],
                    "metadata": document["metadata"],
                },
            )()
        )

        chunks.extend(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in document_chunks
        )

    documents_path = output_dir / "documents.json"
    chunks_path = output_dir / "chunks.json"

    documents_path.write_text(
        json.dumps(
            documents,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    chunks_path.write_text(
        json.dumps(
            chunks,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("CORPUS BUILD COMPLETE")
    print("=" * 60)
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(
        f"Chunk size: {chunking_config['chunk_size']}"
    )
    print(
        f"Chunk overlap: {chunking_config['chunk_overlap']}"
    )
    print(f"Documents saved to: {documents_path}")
    print(f"Chunks saved to: {chunks_path}")


if __name__ == "__main__":
    main()