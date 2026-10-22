from pathlib import Path

from .dense import DenseRetriever
from .qdrant_store import LocalQdrantIndex
from .types import SearchResult


class PersistentDenseRetriever:
    def __init__(self, path: Path, collection_name: str, embedding_model: str):
        self.embedder = DenseRetriever(embedding_model)
        self.index = LocalQdrantIndex(path, collection_name)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_vector = self.embedder.embed([query])[0]
        return self.index.search(query_vector, top_k=top_k)

