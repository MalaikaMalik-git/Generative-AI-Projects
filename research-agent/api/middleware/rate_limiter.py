"""
api/middleware/rate_limiter.py

Per-API-key sliding-window rate limiter.
Default: 10 requests per key per hour.
Configurable via RATE_LIMIT_REQUESTS and RATE_LIMIT_WINDOW_SECONDS in .env
"""
from __future__ import annotations
import os
import time
from typing import Optional
from collections import defaultdict, deque
from fastapi import Header, HTTPException, status
from dotenv import load_dotenv

load_dotenv()

_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
_WINDOW_SECS:  int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600"))

# {api_key: deque of request timestamps}
_windows: dict[str, deque] = defaultdict(deque)


async def check_rate_limit(
    # default=None — missing header is handled by auth middleware first
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    Sliding-window rate limiter. Raises 429 when a key exceeds its quota.
    Auth middleware runs first; if key is None here the request is already rejected.
    """
    if not x_api_key:
        return  # auth dependency already handles this case

    now = time.time()
    window = _windows[x_api_key]

    # Evict timestamps outside the window
    while window and now - window[0] > _WINDOW_SECS:
        window.popleft()

    if len(window) >= _MAX_REQUESTS:
        retry_after = int(_WINDOW_SECS - (now - window[0])) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {_MAX_REQUESTS} requests per "
                f"{_WINDOW_SECS // 3600}h. Retry after {retry_after}s."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    window.append(now)