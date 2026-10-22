from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int = 0

