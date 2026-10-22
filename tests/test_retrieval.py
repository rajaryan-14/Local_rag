from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion


DOCUMENTS = [
    {"chunk_id": "refund", "text": "Refunds are available within thirty days.", "metadata": {}},
    {"chunk_id": "shipping", "text": "Shipping normally takes five business days.", "metadata": {}},
    {"chunk_id": "support", "text": "Contact support by email for account issues.", "metadata": {}},
]


def test_bm25_prefers_matching_document():
    results = BM25Index(DOCUMENTS).search("refunds")
    assert results[0].chunk_id == "refund"


def test_rrf_merges_results_and_preserves_rank_order():
    index = BM25Index(DOCUMENTS)
    first = index.search("refunds")
    second = index.search("shipping")
    third = index.search("support")
    results = reciprocal_rank_fusion([first, second, third], top_k=3)
    assert results[0].chunk_id == "refund"
    assert [result.rank for result in results] == [1, 2, 3]

