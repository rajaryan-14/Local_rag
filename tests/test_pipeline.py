from app.pipeline import RAGPipeline
from app.retrieval.types import SearchResult


class FakeRetriever:
    def search(self, question, top_k):
        return [SearchResult("chunk-1", "Evidence", 1.0)]


class FakeReranker:
    def rerank(self, question, evidence, top_k):
        return evidence[:top_k]


class FakeGenerator:
    def answer(self, question, evidence):
        return "Supported answer [chunk-1]."


class FakeJudge:
    def evaluate(self, question, answer, evidence):
        return "judge-result"


def test_pipeline_connects_all_stages():
    result = RAGPipeline(FakeRetriever(), FakeGenerator(), FakeReranker(), FakeJudge()).run("Question")
    assert result.answer == "Supported answer [chunk-1]."
    assert result.citation_coverage == 1.0
    assert result.faithfulness == "judge-result"


class RepairingGenerator:
    def answer(self, question, evidence):
        return "Supported answer."

    def repair_citations(self, question, answer, evidence):
        return "Supported answer [chunk-1]."


def test_pipeline_repairs_missing_citations():
    result = RAGPipeline(FakeRetriever(), RepairingGenerator()).run("Question")
    assert result.answer.endswith("[chunk-1].")
    assert result.citation_coverage == 1.0

