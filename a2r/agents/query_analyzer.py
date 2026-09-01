from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from a2r.llm.provider import LLMProvider, ModelUnavailable


@dataclass
class QueryAnalysis:
    domain: str
    confidence: float
    reason: str


FALLBACK_KEYWORDS = {
    "billing": ["invoice", "payment", "refund", "credit", "subscription", "price", "annual", "monthly", "tax"],
    "product": ["api", "webhook", "integration", "export", "sla", "limit", "rate", "mobile", "permission"],
    "hr": ["leave", "pto", "holiday", "remote", "conduct", "expense", "performance", "benefits", "onboarding"],
}


def analyze_query(query: str, provider: LLMProvider) -> tuple[QueryAnalysis, bool]:
    prompt = f"Analyze query: {query}"
    try:
        raw = provider.complete(prompt, json_mode=True)
        data = json.loads(raw)
        return QueryAnalysis(data["domain"], float(data["confidence"]), data.get("reason", "")), False
    except (ModelUnavailable, json.JSONDecodeError, KeyError):
        q = query.lower()
        for domain, words in FALLBACK_KEYWORDS.items():
            if any(w in q for w in words):
                return QueryAnalysis(domain, 0.6, "keyword heuristic"), True
        return QueryAnalysis("product", 0.33, "default"), True
