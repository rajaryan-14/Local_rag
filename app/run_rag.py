import argparse
import json
from pathlib import Path

from app.config import settings
from app.generation.ollama_client import OllamaGenerator
from app.evaluation.judge import OllamaJudge
from app.pipeline import RAGPipeline
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.persistent_hybrid import PersistentHybridRetriever
from app.retrieval.reranker import CrossEncoderReranker


def _load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete local RAG pipeline")
    parser.add_argument("question")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--persistent", action="store_true", help="Use the local Qdrant index")
    args = parser.parse_args()

    chunks = _load_chunks(args.chunks)
    if args.persistent:
        retriever = PersistentHybridRetriever(
            chunks,
            settings.root / "data" / "qdrant",
            settings.qdrant_collection,
            settings.embedding_model,
        )
    else:
        retriever = HybridRetriever(chunks, settings.embedding_model)
    reranker = None if args.no_rerank else CrossEncoderReranker(settings.reranker_model)
    generator = OllamaGenerator(settings.ollama_host, settings.generation_model)
    judge = None if args.no_judge else OllamaJudge(settings.ollama_host, settings.generation_model)
    result = RAGPipeline(retriever, generator, reranker, judge).run(
        args.question, top_k=args.top_k, candidate_k=args.candidate_k
    )

    print(f"Answer:\n{result.answer}\n")
    print(f"Citation coverage: {result.citation_coverage:.2f}")
    if result.faithfulness:
        print(f"Faithfulness: {result.faithfulness.score:.2f} ({result.faithfulness.reason})")
        if result.faithfulness.unsupported_claims:
            print(f"Unsupported claims: {result.faithfulness.unsupported_claims}")
    print("\nEvidence:")
    for item in result.evidence:
        print(f"- [{item.chunk_id}] {item.text[:300].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()

