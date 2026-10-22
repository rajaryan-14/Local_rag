from app.retrieval.types import SearchResult


SYSTEM_PROMPT = """You are a careful document question-answering assistant.
Answer only from the supplied evidence. If the evidence does not support an answer,
say that you do not have enough information. Do not invent facts, citations, or sources.
Every factual claim must include one or more citations in the format [chunk_id].
Keep the answer concise and distinguish uncertainty clearly.
"""


def build_grounded_prompt(question: str, evidence: list[SearchResult]) -> str:
    evidence_text = "\n\n".join(
        f"SOURCE [{item.chunk_id}]\n{item.text}" for item in evidence
    )
    return f"""Question:
{question}

Evidence:
{evidence_text or '[No evidence was retrieved]'}

Write a grounded answer with citations. If the evidence is insufficient, say so explicitly.
"""


def build_citation_repair_prompt(question: str, answer: str, evidence: list[SearchResult]) -> str:
    source_ids = ", ".join(f"[{item.chunk_id}]" for item in evidence)
    return f"""Revise the answer below without changing its meaning.
Every factual sentence must end with a citation using one of these exact IDs: {source_ids}.
Do not invent IDs. Return only the revised answer.

Question: {question}
Answer to revise: {answer}
"""

