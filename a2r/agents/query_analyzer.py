from __future__ import annotations

import json
from pydantic import BaseModel, Field

from a2r.llm.provider import DeterministicProvider, LLMProvider, ModelUnavailable


class QueryAnalysis(BaseModel):
    domain: str = Field(pattern="^(billing|product|hr|general)$")
    intent: str = Field(pattern="^(policy_lookup|how_to|troubleshoot|comparison|other)$")
    complexity: str = Field(pattern="^(simple|medium|complex)$")
    entities: list[str] = Field(min_length=1, max_length=5)


def analyze_query(query: str, provider: LLMProvider) -> tuple[QueryAnalysis, bool]:
    prompt = f"""TASK: classify_query
Return only JSON with domain (billing, product, hr, general), intent
(policy_lookup, how_to, troubleshoot, comparison, other), complexity
(simple, medium, complex), and 1-5 entities.\nQUERY: {query}"""
    for attempt in range(2):
        try:
            return QueryAnalysis.model_validate(json.loads(provider.complete(prompt, json_mode=True))), False
        except (ValueError, ModelUnavailable):
            if attempt == 0:
                continue
    fallback = DeterministicProvider()
    return QueryAnalysis.model_validate(json.loads(fallback.complete(prompt, json_mode=True))), True
