import re
from typing import Iterable

from app.retrieval.types import SearchResult


def recall_at_k(results: Iterable[SearchResult], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    found = {result.chunk_id for result in list(results)[:k]}
    return len(found & relevant_ids) / len(relevant_ids)


def precision_at_k(results: Iterable[SearchResult], relevant_ids: set[str], k: int) -> float:
    top = list(results)[:k]
    if not top:
        return 0.0
    return sum(result.chunk_id in relevant_ids for result in top) / len(top)


def reciprocal_rank(results: Iterable[SearchResult], relevant_ids: set[str]) -> float:
    for rank, result in enumerate(results, start=1):
        if result.chunk_id in relevant_ids:
            return 1 / rank
    return 0.0


def cited_chunk_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[([^\]]+)\]", answer))


def citation_coverage(answer: str, evidence: list[SearchResult]) -> float:
    """Fraction of cited IDs that actually exist in the supplied evidence."""
    citations = cited_chunk_ids(answer)
    if not citations:
        return 0.0
    evidence_ids = {result.chunk_id for result in evidence}
    return len(citations & evidence_ids) / len(citations)

