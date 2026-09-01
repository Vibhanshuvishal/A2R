from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = ""


class QueryResponse(BaseModel):
    query_id: str
    session_id: str = ""
    query: str
    answer: str
    answer_source: str
    pipeline_used: str
    domain: str
    confidence: float
    retrieval_confidence: float
    sources: list[str]
    source_badge: str
    source_badge_color: str = "green"
    cache_hit: bool = False
    latency_ms: float = 0.0


class FeedbackRequest(BaseModel):
    query_id: str
    signal: str


class FeedbackResponse(BaseModel):
    acknowledged: bool
    new_weight: float | None = None


class HealthResponse(BaseModel):
    status: str
    mode: str
    model_status: dict
    collections: list[str]
    cache: dict = {}


class SessionCreateRequest(BaseModel):
    title: str = "New Conversation"
    user_id: str = "default_user"


class SessionUpdateTitleRequest(BaseModel):
    title: str


class SessionDetailResponse(BaseModel):
    session: dict
    messages: list[dict]


class CacheStatsResponse(BaseModel):
    cache_size: int
    hits: int
    misses: int
    hit_rate: float
