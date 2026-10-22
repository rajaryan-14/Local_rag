from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A source document before it is split into retrieval chunks."""

    document_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit with traceable source metadata."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

