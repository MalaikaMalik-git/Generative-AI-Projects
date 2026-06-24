"""
api/routers/chat.py

POST /chat       — synchronous, returns full answer when done
GET  /chat/stream — streaming SSE, tokens arrive in real time
"""
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from api.models import ChatRequest, ChatResponse
from api.services.agent_runner import run_agent, stream_agent
from api.middleware.auth import require_api_key
from api.middleware.rate_limiter import check_rate_limit

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Run research agent (synchronous)",
    description="Runs the full research pipeline and returns the answer when complete.",
)
async def chat(
    body: ChatRequest,
    _: None = Depends(require_api_key),
    __: None = Depends(check_rate_limit),
):
    return run_agent(body.question, body.session_id)


@router.get(
    "/chat/stream",
    summary="Run research agent (streaming SSE)",
    description=(
        "Streams status updates and the final answer as Server-Sent Events. "
        "Each event is `data: {type, content}`. Types: status | chunk | done | error."
    ),
)
async def chat_stream(
    question: str = Query(..., min_length=1, max_length=1000),
    session_id: Optional[str] = Query(default=None),
    _: None = Depends(require_api_key),
    __: None = Depends(check_rate_limit),
):
    return StreamingResponse(
        stream_agent(question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )