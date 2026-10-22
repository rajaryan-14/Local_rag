from app.evaluation.judge import OllamaJudge, parse_judge_response
from app.retrieval.types import SearchResult


def test_judge_parser_accepts_fenced_json():
    result = parse_judge_response({"message": {"content": '```json\n{"faithful": false, "score": 0.2, "unsupported_claims": ["x"], "reason": "missing evidence"}\n```'}})
    assert result.faithful is False
    assert result.score == 0.2
    assert result.unsupported_claims == ["x"]


class FakeJudgeClient:
    def chat(self, **kwargs):
        assert kwargs["messages"][0]["role"] == "system"
        return {"message": {"content": '{"faithful": true, "score": 1, "unsupported_claims": [], "reason": "supported"}'}}


def test_ollama_judge_returns_structured_result():
    result = OllamaJudge("http://localhost:11434", "fake", client=FakeJudgeClient()).evaluate(
        "Question", "Answer [chunk-1]", [SearchResult("chunk-1", "Evidence", 1.0)]
    )
    assert result.faithful is True
    assert result.score == 1

