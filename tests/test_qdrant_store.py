from pathlib import Path

from app.retrieval.qdrant_store import LocalQdrantIndex


def test_local_qdrant_round_trip(tmp_path: Path):
    index = LocalQdrantIndex(tmp_path / "qdrant", "test", 3)
    documents = [{"chunk_id": "one", "text": "evidence", "metadata": {"source": "test"}}]
    index.upsert(documents, [[1.0, 0.0, 0.0]])
    results = index.search([1.0, 0.0, 0.0], top_k=1)
    assert results[0].chunk_id == "one"
    assert results[0].metadata["source"] == "test"

