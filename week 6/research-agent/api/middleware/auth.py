"""
api/middleware/auth.py

Checks for a valid API key in the X-API-Key header.
Set ALLOWED_API_KEYS in .env as a comma-separated list.
"""
from __future__ import annotations
import os
from typing import Optional
from fastapi import Header, HTTPException, status, Request
from dotenv import load_dotenv

load_dotenv()

_raw = os.getenv("ALLOWED_API_KEYS", "") or os.getenv("API_KEY", "")
_ALLOWED: set[str] = {k.strip() for k in _raw.split(",") if k.strip()}


async def require_api_key(
    # default=None so FastAPI returns 401 (our error) not 422 (validation error)
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    Raises 401 if key is missing or not in the allowed set.
    If ALLOWED_API_KEYS is empty (dev mode), accepts any non-empty key.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    if _ALLOWED and x_api_key not in _ALLOWED:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )