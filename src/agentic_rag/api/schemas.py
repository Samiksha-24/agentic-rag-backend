"""
schemas.py
==========
Pydantic request/response models for every endpoint. Kept separate from
route handlers so the API contract is easy to read and reuse (e.g. for
generating OpenAPI docs or a typed frontend client).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- chat --- #

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's question.")
    session_id: Optional[str] = Field(
        default=None, description="Conversation/session id. Omit to start a new session."
    )
    research_mode: Optional[bool] = Field(
        default=None, description="Override the server default research mode for this request."
    )


class SourceItem(BaseModel):
    type: str
    label: str
    score: Optional[float] = None
    doc: Optional[str] = None
    page: Optional[int] = None
    preview: Optional[str] = None


class TraceStep(BaseModel):
    icon: str
    title: str
    detail: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceItem]
    trace: List[TraceStep]
    research_mode: bool
    confidence: Optional[float] = None
    total_latency_ms: int
    retrieval_latency_ms: Optional[int] = None
    generation_latency_ms: Optional[int] = None
    active_model: str


class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    sources: Optional[List[SourceItem]] = None
    confidence: Optional[float] = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatHistoryMessage]


# ----------------------------------------------------------- documents --- #

class DocumentInfo(BaseModel):
    name: str
    chunks: int
    indexed_at: str


class UploadResult(BaseModel):
    name: str
    chunks_indexed: int
    status: str


class UploadResponse(BaseModel):
    uploaded: List[UploadResult]
    skipped: List[str] = Field(default_factory=list)
    documents: List[DocumentInfo]


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class DeleteDocumentResponse(BaseModel):
    name: str
    deleted: bool


# ----------------------------------------------------------- analytics --- #

class AnalyticsQueryVolumePoint(BaseModel):
    date: str
    queries: int


class AnalyticsLatencyPoint(BaseModel):
    hour: str
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None


class AnalyticsSessionsPoint(BaseModel):
    day: str
    sessions: int


class AnalyticsResponse(BaseModel):
    documents_indexed: int
    total_chunks: int
    total_queries: int
    active_sessions: int
    avg_response_time_ms: Optional[float] = None
    avg_retrieval_latency_ms: Optional[float] = None
    avg_generation_latency_ms: Optional[float] = None
    avg_confidence: Optional[float] = None
    active_model: str
    active_llm_provider: str
    research_mode_default: bool
    uptime_seconds: float
    query_volume_daily: List[AnalyticsQueryVolumePoint] = Field(default_factory=list)
    latency_by_hour: List[AnalyticsLatencyPoint] = Field(default_factory=list)
    new_sessions_by_day: List[AnalyticsSessionsPoint] = Field(default_factory=list)


# --------------------------------------------------------------- health --- #

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    uptime_seconds: float
    vector_store_ready: bool
    documents_indexed: int
    active_llm_provider: str
    active_model: str
    warnings: List[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
