"""
api/routers/history.py

GET /history/{session_id}  — fetch conversation history for a session
DELETE /history/{session_id} — clear history for a session
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from api.models import HistoryResponse, HistoryEntry
from api.services.agent_runner import get_history, _history
from api.middleware.auth import require_api_key
from datetime import datetime

router = APIRouter(tags=["history"])


@router.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    summary="Get session history",
)
async def history(
    session_id: str,
    _: None = Depends(require_api_key),
):
    entries_raw = get_history(session_id)
    if not entries_raw:
        raise HTTPException(status_code=404, detail=f"No history found for session '{session_id}'")

    entries = [
        HistoryEntry(
            session_id=e["session_id"],
            question=e["question"],
            answer=e["answer"],
            success=e["success"],
            timestamp=e.get("timestamp", datetime.utcnow()),
        )
        for e in entries_raw
    ]

    return HistoryResponse(
        session_id=session_id,
        entries=entries,
        total=len(entries),
    )


@router.delete(
    "/history/{session_id}",
    summary="Clear session history",
)
async def clear_history(
    session_id: str,
    _: None = Depends(require_api_key),
):
    if session_id not in _history:
        raise HTTPException(status_code=404, detail=f"No history found for session '{session_id}'")
    _history.pop(session_id)
    return {"deleted": True, "session_id": session_id}