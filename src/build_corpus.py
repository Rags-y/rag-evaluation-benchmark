import json
from pathlib import Path

import yaml

from rag_eval.chunker import TextChunker
from rag_eval.loader import DocumentLoader


ROOT_DIR = Path(__file__).resolve().parent.parent


def main():
    config_path = ROOT_DIR / "configs" / "default.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_dir = ROOT_DIR / "data" / "raw"
    processed_dir = ROOT_DIR / "data" / "processed"

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    loader = DocumentLoader(raw_dir)

    chunk_config = config["chunking"]

    chunker = TextChunker(
        chunk_size=chunk_config["chunk_size"],
        chunk_overlap=chunk_config["chunk_overlap"],
    )

    all_documents = []
    all_chunks = []

    seen_urls = set()

    for item in config["corpus"]["documents"]:
        url = item["url"]

        if url in seen_urls:
            continue

        seen_urls.add(url)

        print(f"Loading: {item['title']}")

        document = loader.load(
            document_id=item["id"],
            title=item["title"],
            url=url,
        )

        chunks = chunker.chunk_document(document)

        all_documents.append(
            {
                "document_id": document.document_id,
                "title": document.title,
                "source_url": document.source_url,
                "text": document.text,
                "metadata": document.metadata,
            }
        )

        all_chunks.extend(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        )

        print(
            f"  characters: {len(document.text):,}"
        )
        print(
            f"  chunks: {len(chunks)}"
        )

    documents_path = processed_dir / "documents.json"
    chunks_path = processed_dir / "chunks.json"

    documents_path.write_text(
        json.dumps(
            all_documents,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    chunks_path.write_text(
        json.dumps(
            all_chunks,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Corpus build complete.")
    print(f"Documents: {len(all_documents)}")
    print(f"Chunks: {len(all_chunks)}")
    print(f"Saved: {documents_path}")
    print(f"Saved: {chunks_path}")


if __name__ == "__main__":
    main()