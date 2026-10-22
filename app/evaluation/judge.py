import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.types import SearchResult


JUDGE_SYSTEM_PROMPT = """You are a strict faithfulness evaluator for a retrieval-augmented answer.
Judge only whether the answer is supported by the supplied evidence. Do not reward outside
knowledge or plausible guesses. Return JSON only with this schema:
{
  \"faithful\": true,
  \"score\": 0,
  \"unsupported_claims\": [],
  \"reason\": \"...\"
}
Use score 1 for fully supported, 0 for unsupported, and decimals between them for partial support.
"""


@dataclass(frozen=True)
class JudgeResult:
    faithful: bool
    score: float
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str = ""


def build_judge_prompt(question: str, answer: str, evidence: list[SearchResult]) -> str:
    sources = "\n\n".join(f"[{item.chunk_id}] {item.text}" for item in evidence)
    return f"Question: {question}\n\nEvidence:\n{sources or '[none]'}\n\nAnswer:\n{answer}"


def parse_judge_response(response: Any) -> JudgeResult:
    content = response["message"]["content"] if isinstance(response, dict) else response.message.content
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("Judge did not return a JSON object")
    data = json.loads(match.group(0))
    return JudgeResult(
        faithful=bool(data.get("faithful", False)),
        score=float(data.get("score", 0.0)),
        unsupported_claims=list(data.get("unsupported_claims", [])),
        reason=str(data.get("reason", "")),
    )


class OllamaJudge:
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

    def evaluate(self, question: str, answer: str, evidence: list[SearchResult]) -> JudgeResult:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_prompt(question, answer, evidence)},
            ],
        )
        return parse_judge_response(response)

