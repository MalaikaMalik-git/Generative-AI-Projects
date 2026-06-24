"""
api/models.py — Pydantic schemas for all request/response types.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


# ── Requests ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Research question")
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for history tracking. Auto-generated if omitted."
    )

    model_config = {"json_schema_extra": {"example": {"question": "What is quantum computing?"}}}


# ── Responses ───────────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    success: bool
    error: Optional[str] = None
    usage: Optional["TokenUsage"] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    model: str
    agent: str = "research-agent-v6"


class HistoryEntry(BaseModel):
    session_id: str
    question: str
    answer: str
    success: bool
    timestamp: datetime


class HistoryResponse(BaseModel):
    session_id: str
    entries: list[HistoryEntry]
    total: int


# Rebuild for forward reference
ChatResponse.model_rebuild()