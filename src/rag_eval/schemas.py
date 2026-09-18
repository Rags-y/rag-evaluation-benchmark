from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A cleaned source document."""

    document_id: str
    title: str
    source_url: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk created from a source document."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)