from app.generation.ollama_client import OllamaGenerator
from app.generation.prompts import build_grounded_prompt
from app.retrieval.types import SearchResult


class FakeClient:
    def chat(self, **kwargs):
        assert kwargs["messages"][0]["role"] == "system"
        assert "[chunk-1]" in kwargs["messages"][1]["content"]
        return {"message": {"content": "The answer is supported [chunk-1]."}}


def test_prompt_contains_evidence_citation_id():
    prompt = build_grounded_prompt(
        "What is the answer?",
        [SearchResult("chunk-1", "The answer is forty-two.", 1.0)],
    )
    assert "SOURCE [chunk-1]" in prompt
    assert "forty-two" in prompt


def test_ollama_generator_returns_chat_content():
    answer = OllamaGenerator("http://localhost:11434", "fake", client=FakeClient()).answer(
        "What is the answer?",
        [SearchResult("chunk-1", "The answer is forty-two.", 1.0)],
    )
    assert answer == "The answer is supported [chunk-1]."

