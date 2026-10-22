import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.evaluation.judge import OllamaJudge
from app.generation.ollama_client import OllamaGenerator
from app.pipeline import RAGPipeline
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    use_judge: bool = True


class EvidenceItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict = {}


class AskResponse(BaseModel):
    answer: str
    citation_coverage: float
    faithfulness_score: float | None = None
    faithful: bool | None = None
    unsupported_claims: list[str] = []
    evidence: list[EvidenceItem]


app = FastAPI(title="Local RAG API", version="0.1.0")


def _load_chunks() -> list[dict]:
    path = settings.root / "data" / "processed" / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError("Run `py -m app.ingestion.cli` before starting the API")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=1)
def _base_components():
    chunks = _load_chunks()
    return chunks, HybridRetriever(chunks, settings.embedding_model), OllamaGenerator(
        settings.ollama_host, settings.generation_model
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        _, retriever, generator = _base_components()
        reranker = CrossEncoderReranker(settings.reranker_model) if request.use_reranker else None
        judge = OllamaJudge(settings.ollama_host, settings.generation_model) if request.use_judge else None
        result = RAGPipeline(retriever, generator, reranker, judge).run(
            request.question, top_k=request.top_k
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskResponse(
        answer=result.answer,
        citation_coverage=result.citation_coverage,
        faithfulness_score=result.faithfulness.score if result.faithfulness else None,
        faithful=result.faithfulness.faithful if result.faithfulness else None,
        unsupported_claims=result.faithfulness.unsupported_claims if result.faithfulness else [],
        evidence=[EvidenceItem(
            chunk_id=item.chunk_id,
            text=item.text,
            score=item.score,
            metadata=item.metadata,
        ) for item in result.evidence],
    )

