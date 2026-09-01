from __future__ import annotations

import json
from statistics import harmonic_mean
from pydantic import BaseModel, Field

from a2r.llm.provider import DeterministicProvider, LLMProvider, ModelUnavailable


class ValidationScores(BaseModel):
    relevance: float = Field(ge=0, le=1)
    grounding: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)

    @property
    def confidence(self) -> float:
        values = [self.relevance, self.grounding, self.completeness]
        return harmonic_mean(values) if all(values) else 0.0


def validate_answer(query: str, answer: str, chunks: list[dict], provider: LLMProvider) -> tuple[ValidationScores, bool]:
    if not answer or not chunks:
        return ValidationScores(relevance=0, grounding=0, completeness=0), False
    context = "\n".join(chunk["text"][:500] for chunk in chunks[:3])
    prompt = f"""TASK: validate_answer
Return only JSON with relevance, grounding, completeness from 0 to 1.
QUESTION: {query}\nANSWER: {answer}\nCONTEXT: {context}"""
    for attempt in range(2):
        try:
            return ValidationScores.model_validate(json.loads(provider.complete(prompt, json_mode=True))), False
        except (ValueError, ModelUnavailable):
            if attempt == 0:
                continue
    fallback = DeterministicProvider()
    return ValidationScores.model_validate(json.loads(fallback.complete(prompt, json_mode=True))), True
