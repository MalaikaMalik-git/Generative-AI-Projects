"""
api/services/agent_runner.py

Bridge between FastAPI and the Week 6 research agent.
Works regardless of whether api/ lives inside research-agent/ (your setup)
or alongside it (original plan).
"""
from __future__ import annotations
import sys
import os
import asyncio
import time
import json
import uuid
from typing import AsyncGenerator, Optional
from datetime import datetime

# ── Path resolution ─────────────────────────────────────────────────────────────
# api/services/agent_runner.py  →  api/  →  project root
_API_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR = os.path.dirname(_API_DIR)

# Support two layouts:
#   Layout A: research-agent/api/services/agent_runner.py  (your actual setup)
#             _PROJECT_DIR == research-agent/ which already has agent/ inside it
#   Layout B: week10/api/services/agent_runner.py
#             _PROJECT_DIR == week10/, research-agent/ is a sibling folder

def _ensure_agent_importable() -> None:
    # Layout A: agent/ is directly in _PROJECT_DIR
    if os.path.isdir(os.path.join(_PROJECT_DIR, "agent")):
        if _PROJECT_DIR not in sys.path:
            sys.path.insert(0, _PROJECT_DIR)
        return
    # Layout B: look for research-agent/ sibling
    sibling = os.path.join(_PROJECT_DIR, "research-agent")
    if os.path.isdir(os.path.join(sibling, "agent")):
        if sibling not in sys.path:
            sys.path.insert(0, sibling)
        return
    raise ImportError(
        f"Cannot find 'agent/' package. Looked in:\n"
        f"  {_PROJECT_DIR}\n  {sibling}\n"
        "Make sure you run uvicorn from inside the research-agent folder."
    )

_ensure_agent_importable()

from agent.decomposer    import decompose       # noqa: E402
from agent.react_loop    import run             # noqa: E402
from agent.error_handler import safe_synthesize # noqa: E402
from api.models          import ChatResponse, TokenUsage
from api.services.cost_logger import CostLogger

# ── State ───────────────────────────────────────────────────────────────────────
_history: dict[str, list[dict]] = {}
_cost_logger = CostLogger()

# Cost log file — written next to this file's project root
_COST_LOG_PATH = os.path.join(_PROJECT_DIR, "cost_log.jsonl")


def _new_session() -> str:
    return str(uuid.uuid4())


def _store(session_id: str, entry: dict) -> None:
    _history.setdefault(session_id, []).append(entry)


def _write_cost_log(session_id: str, question: str, usage: TokenUsage) -> None:
    """Append one line to cost_log.jsonl for auditing."""
    try:
        record = {
            "ts":          datetime.utcnow().isoformat(),
            "session_id":  session_id,
            "question":    question[:120],
            "tokens":      usage.total_tokens,
            "cost_usd":    usage.estimated_cost_usd,
        }
        with open(_COST_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # never crash the request over logging


def get_history(session_id: str) -> list[dict]:
    return _history.get(session_id, [])


# ── Sync runner ─────────────────────────────────────────────────────────────────

def run_agent(question: str, session_id: Optional[str] = None) -> ChatResponse:
    """Run full research pipeline synchronously."""
    session_id = session_id or _new_session()

    try:
        try:
            sub_questions = decompose(question)
        except Exception:
            sub_questions = [question]

        sub_results = []
        for sub_q in sub_questions:
            result = run(sub_q, verbose=False)
            sub_results.append({
                "question": sub_q,
                "answer":   result.answer,
                "success":  result.success,
            })

        report = safe_synthesize(question, sub_results)
        usage  = _cost_logger.estimate(question, report, sub_questions)
        _write_cost_log(session_id, question, usage)

        response = ChatResponse(
            session_id=session_id,
            question=question,
            answer=report,
            success=True,
            usage=usage,
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        response = ChatResponse(
            session_id=session_id,
            question=question,
            answer=f"Agent pipeline failed: {str(e)}",
            success=False,
            error=str(e),
            timestamp=datetime.utcnow(),
        )

    _store(session_id, response.model_dump(mode="json"))
    return response


# ── Streaming runner ─────────────────────────────────────────────────────────────

async def stream_agent(
    question: str,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream status + answer as SSE.
    Event format: data: {"type": "status"|"chunk"|"done"|"error", ...}
    """
    session_id = session_id or _new_session()

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    yield _sse({"type": "status", "content": "🔍 Decomposing question..."})

    try:
        loop = asyncio.get_event_loop()

        try:
            sub_questions = await loop.run_in_executor(None, decompose, question)
        except Exception:
            sub_questions = [question]

        yield _sse({"type": "status", "content": f"📋 {len(sub_questions)} sub-questions identified"})

        sub_results = []
        for i, sub_q in enumerate(sub_questions, 1):
            yield _sse({"type": "status", "content": f"🔎 [{i}/{len(sub_questions)}] {sub_q[:80]}"})
            result = await loop.run_in_executor(None, lambda q=sub_q: run(q, verbose=False))
            sub_results.append({
                "question": sub_q,
                "answer":   result.answer,
                "success":  result.success,
            })

        yield _sse({"type": "status", "content": "✍️ Synthesizing final report..."})

        report = await loop.run_in_executor(None, safe_synthesize, question, sub_results)

        # Stream report in word chunks for real-time feel
        words = report.split(" ")
        buf: list[str] = []
        for word in words:
            buf.append(word)
            if len(buf) >= 8:
                yield _sse({"type": "chunk", "content": " ".join(buf) + " "})
                buf = []
                await asyncio.sleep(0.015)
        if buf:
            yield _sse({"type": "chunk", "content": " ".join(buf)})

        usage = _cost_logger.estimate(question, report, sub_questions)
        _write_cost_log(session_id, question, usage)

        _store(session_id, ChatResponse(
            session_id=session_id,
            question=question,
            answer=report,
            success=True,
            usage=usage,
            timestamp=datetime.utcnow(),
        ).model_dump(mode="json"))

        yield _sse({
            "type":       "done",
            "session_id": session_id,
            "success":    True,
            "usage":      usage.model_dump(),
        })

    except Exception as e:
        yield _sse({"type": "error",  "content": str(e)})
        yield _sse({"type": "done",   "session_id": session_id, "success": False})