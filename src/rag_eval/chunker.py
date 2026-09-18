from .schemas import Chunk, Document


class TextChunker:
    """Split documents into overlapping character-based chunks."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split one document into overlapping chunks."""

        text = document.text

        if not text.strip():
            return []

        chunks = []

        start = 0
        chunk_number = 0

        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = (
                    f"{document.document_id}"
                    f"-chunk-{chunk_number:04d}"
                )

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        text=chunk_text,
                        metadata={
                            "chunk_number": chunk_number,
                            "start_char": start,
                            "end_char": end,
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                            "source_url": document.source_url,
                            "title": document.title,
                        },
                    )
                )

                chunk_number += 1

            start += step

        return chunks