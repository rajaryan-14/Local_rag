from pathlib import Path

from .models import Document


def load_text_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        text = BeautifulSoup(text, "html.parser").get_text("\n")
    return Document(
        document_id=path.stem,
        source=str(path),
        text=text,
        metadata={"filename": path.name, "extension": path.suffix.lower()},
    )


def load_pdf_file(path: Path) -> Document:
    import fitz

    pdf = fitz.open(path)
    pages = [page.get_text("text") for page in pdf]
    return Document(
        document_id=path.stem,
        source=str(path),
        text="\n\n".join(pages),
        metadata={"filename": path.name, "extension": ".pdf", "page_count": len(pages)},
    )


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_file(path)
    if suffix in {".txt", ".md", ".markdown", ".html", ".htm"}:
        return load_text_file(path)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def load_directory(directory: Path) -> list[Document]:
    supported = {".pdf", ".txt", ".md", ".markdown", ".html", ".htm"}
    return [
        load_document(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in supported
    ]

