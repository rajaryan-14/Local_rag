from dataclasses import dataclass
from typing import Any

from app.evaluation.judge import JudgeResult
from app.evaluation.metrics import citation_coverage
from app.retrieval.types import SearchResult


@dataclass(frozen=True)
class PipelineResult:
    question: str
    evidence: list[SearchResult]
    answer: str
    citation_coverage: float
    faithfulness: JudgeResult | None = None


class RAGPipeline:
    def __init__(self, retriever: Any, generator: Any, reranker: Any | None = None, judge: Any | None = None):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.judge = judge

    def run(self, question: str, top_k: int = 5, candidate_k: int = 20) -> PipelineResult:
        evidence = self.retriever.search(question, top_k=candidate_k)
        if self.reranker is not None:
            evidence = self.reranker.rerank(question, evidence, top_k=top_k)
        else:
            evidence = evidence[:top_k]

        answer = self.generator.answer(question, evidence)
        coverage = citation_coverage(answer, evidence)
        if coverage < 1.0 and hasattr(self.generator, "repair_citations"):
            repaired = self.generator.repair_citations(question, answer, evidence)
            if citation_coverage(repaired, evidence) > coverage:
                answer = repaired
                coverage = citation_coverage(answer, evidence)
        faithfulness = self.judge.evaluate(question, answer, evidence) if self.judge else None
        return PipelineResult(
            question=question,
            evidence=evidence,
            answer=answer,
            citation_coverage=coverage,
            faithfulness=faithfulness,
        )

