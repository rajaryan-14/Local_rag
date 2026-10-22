from pathlib import Path

from .types import SearchResult


def _as_list(vector):
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)


class LocalQdrantIndex:
    """Persistent dense index using Qdrant's embedded local mode."""

    def __init__(self, path: Path, collection_name: str, vector_size: int | None = None):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError("Install qdrant-client to use the persistent index.") from exc

        self.models = models
        self.client = QdrantClient(path=str(path))
        self.collection_name = collection_name
        if not self.client.collection_exists(collection_name):
            if vector_size is None:
                raise ValueError(f"Qdrant collection does not exist: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def upsert(self, documents: list[dict], vectors) -> None:
        points = [
            self.models.PointStruct(
                id=index,
                vector=_as_list(vector),
                payload={
                    "chunk_id": document["chunk_id"],
                    "text": document["text"],
                    "metadata": document.get("metadata", {}),
                },
            )
            for index, (document, vector) in enumerate(zip(documents, vectors))
        ]
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, top_k: int = 10) -> list[SearchResult]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=_as_list(query_vector),
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchResult(
                chunk_id=point.payload["chunk_id"],
                text=point.payload["text"],
                score=float(point.score),
                metadata=point.payload.get("metadata", {}),
                rank=rank,
            )
            for rank, point in enumerate(response.points, start=1)
        ]

