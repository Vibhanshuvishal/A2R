from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="", max_length=128)

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class QueryResponse(BaseModel):
    answer: str
    answer_source: str
    pipeline_used: str | None = None
    domain: str
    confidence: float
    retrieval_confidence: float = 0.0
    sources: list[str] = []
    query_id: str
    session_id: str = ""
    source_badge: str
    source_badge_color: str
    needs_human_review: bool = False
    used_fallback: bool = False
    cache_hit: bool = False
    latency_ms: float = 0.0


class FeedbackRequest(BaseModel):
    query_id: str
    signal: Literal["accept", "reject"]


class FeedbackResponse(BaseModel):
    acknowledged: bool
    new_weight: float | None = None


class HealthResponse(BaseModel):
    status: str
    runtime_mode: Literal["local", "public_demo"]
    model_status: dict
    cache: dict[str, Any] = {}


class SessionCreateRequest(BaseModel):
    title: str = "New Conversation"
    user_id: str = "default"


class SessionUpdateTitleRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)


class MessageItem(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    query_id: str = ""
    metadata: dict[str, Any] = {}
    timestamp: float


class SessionDetailResponse(BaseModel):
    session: dict[str, Any]
    messages: list[MessageItem]


class CacheStatsResponse(BaseModel):
    cache_size: int
    hits: int
    misses: int
    total_lookups: int
    hit_rate: float
    threshold: float
