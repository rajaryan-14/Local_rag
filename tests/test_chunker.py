from app.ingestion.chunker import chunk_document
from app.ingestion.models import Document


def test_chunker_preserves_traceable_ids_and_overlap():
    document = Document("demo", "demo.txt", "First paragraph.\n\n" + ("important text " * 120))
    chunks = chunk_document(document, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "demo_chunk_0000"
    assert all(chunk.document_id == "demo" for chunk in chunks)
    assert all(chunk.metadata["source"] == "demo.txt" for chunk in chunks)

