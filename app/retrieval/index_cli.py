import argparse
import json
from pathlib import Path

from app.config import settings
from .dense import DenseRetriever
from .qdrant_store import LocalQdrantIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the persistent local Qdrant dense index")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--path", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--collection", default=settings.qdrant_collection)
    args = parser.parse_args()

    with args.chunks.open(encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle if line.strip()]
    embedder = DenseRetriever(settings.embedding_model)
    vectors = embedder.embed([document["text"] for document in documents])
    index = LocalQdrantIndex(args.path, args.collection, vectors.shape[1])
    index.upsert(documents, vectors)
    print(f"Indexed {len(documents)} chunks in {args.path}")


if __name__ == "__main__":
    main()

