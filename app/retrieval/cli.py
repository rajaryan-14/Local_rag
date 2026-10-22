import argparse
import json
from pathlib import Path

from app.config import settings
from .bm25 import BM25Index
from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .hybrid import HybridRetriever
from .reranker import CrossEncoderReranker


def _load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local retrieval pipeline")
    parser.add_argument("query")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    chunks = _load_chunks(args.chunks)
    candidate_k = max(args.top_k, 10)
    if args.mode == "bm25":
        candidates = BM25Index(chunks).search(args.query, top_k=candidate_k)
    elif args.mode == "dense":
        candidates = DenseRetriever(settings.embedding_model).search(
            args.query, chunks, top_k=candidate_k
        )
    else:
        candidates = HybridRetriever(chunks, settings.embedding_model).search(
            args.query, top_k=candidate_k, candidate_k=candidate_k
        )
    if args.rerank:
        candidates = CrossEncoderReranker(settings.reranker_model).rerank(
            args.query, candidates, top_k=args.top_k
        )
    else:
        candidates = candidates[:args.top_k]

    for result in candidates:
        print(f"[{result.rank}] {result.chunk_id} score={result.score:.4f}")
        print(result.text.replace("\n", " ")[:500])
        print()


if __name__ == "__main__":
    main()

