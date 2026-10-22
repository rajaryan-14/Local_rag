import argparse
import json
from pathlib import Path

from .chunker import chunk_document
from .loaders import load_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and chunk local RAG documents")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    documents = load_directory(args.input)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, args.chunk_size, args.overlap)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }, ensure_ascii=False) + "\n")

    print(f"Loaded {len(documents)} documents and wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()

