from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Vector store for semantic retrieval."""

    def __init__(
    self,
    model_name: str = "all-MiniLM-L6-v2",
):
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        """Create embeddings for all chunks."""

        if not chunks:
            raise ValueError("Cannot build index from empty chunks")

        self.chunks = chunks

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self.embeddings = np.asarray(
            self.embeddings,
            dtype=np.float32,
        )

    def save(self, index_dir: str | Path) -> None:
        """Save embeddings and chunk metadata."""

        if self.embeddings is None:
            raise RuntimeError("Vector index has not been built")

        index_dir = Path(index_dir)
        index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            index_dir / "embeddings.npy",
            self.embeddings,
        )

        (index_dir / "chunks.json").write_text(
            json.dumps(
                self.chunks,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(self, index_dir: str | Path) -> None:
        """Load a previously built vector index."""

        index_dir = Path(index_dir)

        embeddings_path = index_dir / "embeddings.npy"
        chunks_path = index_dir / "chunks.json"

        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Missing embeddings: {embeddings_path}"
            )

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Missing chunks: {chunks_path}"
            )

        self.embeddings = np.load(
            embeddings_path
        )

        self.chunks = json.loads(
            chunks_path.read_text(
                encoding="utf-8"
            )
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Return the most similar chunks for a query."""

        if self.embeddings is None:
            raise RuntimeError(
                "Vector index has not been built or loaded"
            )

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            chunk = self.chunks[index].copy()
            chunk["score"] = float(scores[index])
            results.append(chunk)

        return results