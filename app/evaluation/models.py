from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    question: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    reference_answer: str = ""

