from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.types import SearchResult


class FakeCrossEncoder:
    def predict(self, pairs):
        return [len(text) for _, text in pairs]


def test_reranker_sorts_by_model_score_and_resets_ranks():
    candidates = [
        SearchResult("short", "short", 0.1, {}, 1),
        SearchResult("long", "a much longer candidate", 0.2, {}, 2),
    ]
    results = CrossEncoderReranker("fake", model=FakeCrossEncoder()).rerank("query", candidates)

    assert [result.chunk_id for result in results] == ["long", "short"]
    assert [result.rank for result in results] == [1, 2]

