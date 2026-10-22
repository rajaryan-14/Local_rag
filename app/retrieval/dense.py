from .types import SearchResult


class DenseRetriever:
    """Sentence Transformers adapter; indexing storage is intentionally separate."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements.txt to use dense retrieval."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, documents: list[dict], top_k: int = 10) -> list[SearchResult]:
        query_vector = self.embed([query])[0]
        document_vectors = self.embed([doc["text"] for doc in documents])
        scores = document_vectors @ query_vector
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        return [
            SearchResult(
                chunk_id=documents[index]["chunk_id"],
                text=documents[index]["text"],
                score=float(score),
                metadata=documents[index].get("metadata", {}),
                rank=rank,
            )
            for rank, (index, score) in enumerate(ranked[:top_k], start=1)
        ]

