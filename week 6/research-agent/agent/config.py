"""
agent/config.py
Central settings — loaded once at startup from .env
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example → .env and fill in your keys."
        )
    return val


# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = _require("OPENAI_API_KEY")
OPENAI_MODEL: str   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Search ─────────────────────────────────────────────────────────────────────
# DuckDuckGo via `ddgs` — no API key needed, nothing to configure.

# ── Agent behaviour ────────────────────────────────────────────────────────────
MAX_STEPS: int          = int(os.getenv("MAX_STEPS", "6"))
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
REQUEST_TIMEOUT: int    = int(os.getenv("REQUEST_TIMEOUT", "5"))