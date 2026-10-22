from .bm25 import BM25Index
from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .types import SearchResult


class HybridRetriever:
    def __init__(self, documents: list[dict], embedding_model: str):
        self.documents = documents
        self.bm25 = BM25Index(documents)
        self.dense = DenseRetriever(embedding_model)

    def search(self, query: str, top_k: int = 10, candidate_k: int = 20) -> list[SearchResult]:
        sparse_results = self.bm25.search(query, top_k=candidate_k)
        dense_results = self.dense.search(query, self.documents, top_k=candidate_k)
        return reciprocal_rank_fusion([sparse_results, dense_results], top_k=top_k)

