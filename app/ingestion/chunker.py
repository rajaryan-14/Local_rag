import re

from .models import Chunk, Document


def _normalize(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_document(
    document: Document,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Split text by approximate characters while preserving source identity."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")

    text = _normalize(document.text)
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + (1 if text[boundary] == "\n" else 2)

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}_chunk_{index:04d}",
                    document_id=document.document_id,
                    text=chunk_text,
                    metadata={**document.metadata, "source": document.source, "char_start": start, "char_end": end},
                )
            )
            index += 1

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks

