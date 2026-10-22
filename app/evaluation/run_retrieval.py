import argparse
import json
from pathlib import Path

from app.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from app.evaluation.models import EvalCase
from app.retrieval.bm25 import BM25Index


def load_cases(path: Path) -> list[EvalCase]:
    with path.open(encoding="utf-8") as handle:
        return [EvalCase(**json.loads(line)) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the BM25 retrieval baseline")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/golden.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    with args.chunks.open(encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle if line.strip()]
    index = BM25Index(documents)
    cases = load_cases(args.dataset)
    scores = []
    for case in cases:
        results = index.search(case.question, top_k=args.k)
        relevant = set(case.relevant_chunk_ids)
        scores.append({
            "question": case.question,
            "recall": recall_at_k(results, relevant, args.k),
            "precision": precision_at_k(results, relevant, args.k),
            "mrr": reciprocal_rank(results, relevant),
        })

    if not scores:
        print("No evaluation cases found")
        return
    print(json.dumps({
        "cases": scores,
        "mean_recall": sum(item["recall"] for item in scores) / len(scores),
        "mean_precision": sum(item["precision"] for item in scores) / len(scores),
        "mean_mrr": sum(item["mrr"] for item in scores) / len(scores),
    }, indent=2))


if __name__ == "__main__":
    main()

