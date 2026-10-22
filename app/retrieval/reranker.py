from typing import Any

from .types import SearchResult


class CrossEncoderReranker:
    """Rerank a small candidate set with a Sentence Transformers Cross-Encoder."""

    def __init__(self, model_name: str, model: Any | None = None):
        if model is not None:
            self.model = model
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements.txt to use the Cross-Encoder reranker."
            ) from exc
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            SearchResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=float(score),
                metadata=candidate.metadata,
                rank=rank,
            )
            for rank, (candidate, score) in enumerate(ranked[:top_k], start=1)
        ]

