import math
import re
from collections import Counter

from .types import SearchResult


def _tokens(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w+\b", text.lower())


class BM25Index:
    """Small dependency-free BM25 index for local development and tests."""

    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(_tokens(doc["text"])) for doc in documents]
        self.doc_lengths = [sum(freq.values()) for freq in self.term_frequencies]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_freq: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            self.doc_freq.update(frequencies.keys())

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_terms = _tokens(query)
        scored: list[tuple[int, float]] = []
        n_docs = len(self.documents)
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            for term in query_terms:
                if term not in frequencies:
                    continue
                df = self.doc_freq[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                tf = frequencies[term]
                length_norm = 1 - self.b + self.b * self.doc_lengths[index] / max(self.avgdl, 1)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
            if score > 0:
                scored.append((index, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        results = []
        for rank, (index, score) in enumerate(scored[:top_k], start=1):
            doc = self.documents[index]
            results.append(SearchResult(
                chunk_id=doc["chunk_id"],
                text=doc["text"],
                score=score,
                metadata=doc.get("metadata", {}),
                rank=rank,
            ))
        return results

