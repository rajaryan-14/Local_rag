from pathlib import Path

from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion
from .persistent import PersistentDenseRetriever
from .types import SearchResult


class PersistentHybridRetriever:
    def __init__(self, documents: list[dict], qdrant_path: Path, collection_name: str, embedding_model: str):
        self.bm25 = BM25Index(documents)
        self.dense = PersistentDenseRetriever(qdrant_path, collection_name, embedding_model)

    def search(self, query: str, top_k: int = 10, candidate_k: int = 20) -> list[SearchResult]:
        sparse = self.bm25.search(query, top_k=candidate_k)
        dense = self.dense.search(query, top_k=candidate_k)
        return reciprocal_rank_fusion([sparse, dense], top_k=top_k)

