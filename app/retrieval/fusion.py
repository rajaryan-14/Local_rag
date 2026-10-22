from collections import defaultdict

from .types import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    top_k: int = 10,
    smoothing: int = 60,
) -> list[SearchResult]:
    """Merge ranked lists without requiring scores to be comparable."""
    scores: defaultdict[str, float] = defaultdict(float)
    representative: dict[str, SearchResult] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] += 1 / (smoothing + rank)
            representative.setdefault(result.chunk_id, result)

    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
    return [
        SearchResult(
            chunk_id=chunk_id,
            text=representative[chunk_id].text,
            score=scores[chunk_id],
            metadata=representative[chunk_id].metadata,
            rank=rank,
        )
        for rank, chunk_id in enumerate(ranked_ids, start=1)
    ]

