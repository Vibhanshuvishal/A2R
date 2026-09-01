from dataclasses import dataclass
from typing import Any
from a2r.llm.provider import LLMProvider


@dataclass
class ValidationResult:
    is_grounded: bool
    confidence: float
    reason: str


def validate_answer(answer: str, retrieved_chunks: list[dict[str, Any]], provider: LLMProvider) -> ValidationResult:
    if not retrieved_chunks:
        return ValidationResult(False, 0.0, "no retrieved chunks provided")
    lower = answer.lower()
    if "outside the indexed knowledge base" in lower or "does not contain" in lower:
        return ValidationResult(False, 0.2, "model flagged missing evidence")
    return ValidationResult(True, 0.9, "answer accepted")
