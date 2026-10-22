from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    root: Path = Path(__file__).resolve().parents[1]
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_documents")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    generation_model: str = os.getenv("OLLAMA_GENERATION_MODEL", "qwen2.5vl:3b")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")

    @property
    def raw_data_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def eval_data_dir(self) -> Path:
        return self.root / "data" / "eval"


settings = Settings()
