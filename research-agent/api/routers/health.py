"""
api/routers/health.py — GET /health
"""
from fastapi import APIRouter
from api.models import HealthResponse
import os

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        agent="research-agent-v6",
    )