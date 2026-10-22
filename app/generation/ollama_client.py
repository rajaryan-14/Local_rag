from typing import Any

from app.retrieval.types import SearchResult
from .prompts import SYSTEM_PROMPT, build_citation_repair_prompt, build_grounded_prompt


class OllamaGenerator:
    def __init__(self, host: str, model: str, client: Any | None = None):
        if client is not None:
            self.client = client
        else:
            try:
                from ollama import Client
            except ImportError as exc:
                raise RuntimeError("Install the ollama Python package first.") from exc
            self.client = Client(host=host)
        self.model = model

    def answer(self, question: str, evidence: list[SearchResult]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_grounded_prompt(question, evidence)},
            ],
        )
        if isinstance(response, dict):
            return response["message"]["content"]
        return response.message.content

    def repair_citations(self, question: str, answer: str, evidence: list[SearchResult]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_citation_repair_prompt(question, answer, evidence)},
            ],
        )
        if isinstance(response, dict):
            return response["message"]["content"]
        return response.message.content

