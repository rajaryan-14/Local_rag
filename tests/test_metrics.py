from app.evaluation.metrics import citation_coverage, precision_at_k, recall_at_k, reciprocal_rank
from app.retrieval.types import SearchResult


RESULTS = [SearchResult("a", "A", 1.0), SearchResult("b", "B", 0.5), SearchResult("c", "C", 0.2)]


def test_retrieval_metrics():
    relevant = {"b", "c"}
    assert recall_at_k(RESULTS, relevant, 2) == 0.5
    assert precision_at_k(RESULTS, relevant, 2) == 0.5
    assert reciprocal_rank(RESULTS, relevant) == 0.5


def test_citation_coverage_rejects_unknown_ids():
    assert citation_coverage("Supported [a] but not [unknown].", RESULTS) == 0.5
    assert citation_coverage("No citation", RESULTS) == 0.0

