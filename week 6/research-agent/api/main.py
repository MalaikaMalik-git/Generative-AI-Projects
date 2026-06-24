"""
api/main.py — FastAPI application entry point.

Registers all routers, configures CORS and startup logging.
"""
from __future__ import annotations
import os
import sys

# Ensure research-agent is importable from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_AGENT = os.path.join(_ROOT, "research-agent")
for p in (_ROOT, _AGENT):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import health, chat, history

# ── App ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Research Agent API",
    description=(
        "A production-structured FastAPI backend wrapping the Week 6 ReAct research agent. "
        "Streaming enabled. Auth via X-API-Key header. Rate limited per session."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ────────────────────────────────────────────────────────────────────────
# FRONTEND_URL is set in production (e.g. https://your-app.vercel.app)
# Falls back to wildcard for local dev if not set.
_frontend_url = os.getenv("FRONTEND_URL", "")
_origins = [_frontend_url] if _frontend_url else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)

# ── Routers ─────────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(history.router)

# ── Root ────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name":    "Research Agent API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }