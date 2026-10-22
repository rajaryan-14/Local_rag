from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.types import SearchResult


def test_hybrid_fusion_promotes_documents_found_by_both_retrievers():
    sparse = [
        SearchResult("shared", "shared evidence", 5.0),
        SearchResult("sparse_only", "keyword evidence", 4.0),
    ]
    dense = [
        SearchResult("shared", "shared evidence", 0.9),
        SearchResult("dense_only", "semantic evidence", 0.8),
    ]

    results = reciprocal_rank_fusion([sparse, dense], top_k=3)

    assert results[0].chunk_id == "shared"
    assert {result.chunk_id for result in results} == {"shared", "sparse_only", "dense_only"}

