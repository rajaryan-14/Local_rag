from pathlib import Path

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.types import SearchResult


def test_persistent_hybrid_fusion_contract():
    sparse = [SearchResult("one", "evidence", 1.0)]
    dense = [SearchResult("one", "evidence", 0.9)]
    result = reciprocal_rank_fusion([sparse, dense], top_k=1)
    assert result[0].chunk_id == "one"

