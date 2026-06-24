"""
tests/test_api.py

Full API test suite — no real OpenAI or DuckDuckGo calls.
Run from the research-agent folder:
    pytest tests/test_api.py -v
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ── Path resolution ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_AGENT_PARENT = _ROOT
if not os.path.isdir(os.path.join(_ROOT, "agent")):
    _candidate = os.path.join(_ROOT, "research-agent")
    if os.path.isdir(os.path.join(_candidate, "agent")):
        _AGENT_PARENT = _candidate

for _p in (_ROOT, _AGENT_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Env — set BEFORE any module import ───────────────────────────────────────────
os.environ["OPENAI_API_KEY"]           = "test-fake-key"
os.environ["ALLOWED_API_KEYS"]         = "test-key-good"
os.environ["RATE_LIMIT_REQUESTS"]      = "5"
os.environ["RATE_LIMIT_WINDOW_SECONDS"]= "3600"

# ── Mocks ─────────────────────────────────────────────────────────────────────────
_MOCK_REPORT = "## Summary\nQuantum computers use qubits.\n\n## Sources\n1. https://example.com"

def _fake_run(question, verbose=False):
    r = MagicMock()
    r.answer  = "Quantum computing uses qubits."
    r.success = True
    return r

with patch("agent.decomposer.decompose",         return_value=["sub q1", "sub q2"]), \
     patch("agent.react_loop.run",               side_effect=_fake_run), \
     patch("agent.error_handler.safe_synthesize", return_value=_MOCK_REPORT):
    from api.main import app

client = TestClient(app, raise_server_exceptions=False)

GOOD_KEY = "test-key-good"
BAD_KEY  = "bad-key-000"

# ── Fixture: reset rate-limit window between tests ────────────────────────────────
@pytest.fixture(autouse=True)
def reset_rate_windows():
    from api.middleware.rate_limiter import _windows
    _windows.clear()
    yield
    _windows.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────────
def _patched_chat(question="What is quantum computing?", key=GOOD_KEY, session_id=None):
    with patch("api.services.agent_runner.decompose",         return_value=["sub q"]), \
         patch("api.services.agent_runner.run",               side_effect=_fake_run), \
         patch("api.services.agent_runner.safe_synthesize",   return_value=_MOCK_REPORT):
        body = {"question": question}
        if session_id:
            body["session_id"] = session_id
        return client.post("/chat", json=body, headers={"X-API-Key": key})


# ── Health ────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["agent"] == "research-agent-v6"

    def test_no_auth_required(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_root_has_docs_link(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "docs" in r.json()


# ── Auth ──────────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_missing_key_returns_401(self):
        r = client.post("/chat", json={"question": "test"})
        assert r.status_code == 401

    def test_wrong_key_returns_401(self):
        r = _patched_chat(key=BAD_KEY)
        assert r.status_code == 401

    def test_valid_key_passes(self):
        r = _patched_chat(key=GOOD_KEY)
        assert r.status_code == 200

    def test_stream_missing_key_returns_401(self):
        r = client.get("/chat/stream", params={"question": "test"})
        assert r.status_code == 401

    def test_stream_wrong_key_returns_401(self):
        r = client.get("/chat/stream", params={"question": "test"}, headers={"X-API-Key": BAD_KEY})
        assert r.status_code == 401

    def test_history_missing_key_returns_401(self):
        r = client.get("/history/any-session")
        assert r.status_code == 401


# ── Rate Limiter ──────────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_rate_limit_triggers_after_limit(self):
        limit = int(os.environ["RATE_LIMIT_REQUESTS"])
        hit_429 = False
        for i in range(limit + 1):
            r = _patched_chat()   # same key each time; fixture resets between test methods
            if r.status_code == 429:
                hit_429 = True
                assert "Retry-After" in r.headers
                break
        assert hit_429, f"Expected 429 after {limit} requests"

    def test_rate_limit_error_message(self):
        limit = int(os.environ["RATE_LIMIT_REQUESTS"])
        for _ in range(limit):
            _patched_chat()
        r = _patched_chat()
        assert r.status_code == 429
        assert "Rate limit exceeded" in r.json()["detail"]

    def test_different_keys_have_separate_buckets(self):
        """Two different API keys should have independent limits."""
        from api.middleware.rate_limiter import _windows
        # Exhaust key A
        limit = int(os.environ["RATE_LIMIT_REQUESTS"])
        for _ in range(limit):
            _patched_chat(key=GOOD_KEY)
        # Key A is now rate-limited
        r_a = _patched_chat(key=GOOD_KEY)
        assert r_a.status_code == 429
        # Key B (not in ALLOWED_API_KEYS → 401, but it reaches rate limiter logic separately)
        # Use a second allowed key by patching _ALLOWED temporarily
        import api.middleware.auth as auth_mod
        orig = auth_mod._ALLOWED.copy()
        auth_mod._ALLOWED.add("second-key-ok")
        try:
            r_b = _patched_chat(key="second-key-ok")
            assert r_b.status_code == 200  # fresh bucket, not rate-limited
        finally:
            auth_mod._ALLOWED = orig


# ── Chat (sync) ───────────────────────────────────────────────────────────────────

class TestChat:
    def test_returns_200(self):
        assert _patched_chat().status_code == 200

    def test_has_answer(self):
        assert len(_patched_chat().json()["answer"]) > 0

    def test_has_session_id(self):
        assert len(_patched_chat().json()["session_id"]) > 0

    def test_has_usage(self):
        usage = _patched_chat().json()["usage"]
        assert usage is not None
        assert usage["total_tokens"] > 0
        assert "estimated_cost_usd" in usage

    def test_session_id_preserved(self):
        sid = "my-custom-session-42"
        assert _patched_chat(session_id=sid).json()["session_id"] == sid

    def test_success_flag_true(self):
        assert _patched_chat().json()["success"] is True

    def test_empty_question_rejected(self):
        r = client.post("/chat", json={"question": ""}, headers={"X-API-Key": GOOD_KEY})
        assert r.status_code == 422

    def test_question_too_long_rejected(self):
        r = client.post("/chat", json={"question": "x" * 1001}, headers={"X-API-Key": GOOD_KEY})
        assert r.status_code == 422


# ── History ───────────────────────────────────────────────────────────────────────

class TestHistory:
    def _sid(self) -> str:
        return _patched_chat().json()["session_id"]

    def test_history_returns_entries(self):
        sid = self._sid()
        r = client.get(f"/history/{sid}", headers={"X-API-Key": GOOD_KEY})
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == sid
        assert data["total"] >= 1

    def test_entry_has_required_fields(self):
        sid = self._sid()
        r = client.get(f"/history/{sid}", headers={"X-API-Key": GOOD_KEY})
        entry = r.json()["entries"][0]
        for field in ("question", "answer", "success", "timestamp"):
            assert field in entry

    def test_missing_session_404(self):
        r = client.get("/history/does-not-exist-xyz", headers={"X-API-Key": GOOD_KEY})
        assert r.status_code == 404

    def test_delete_clears_history(self):
        sid = self._sid()
        client.delete(f"/history/{sid}", headers={"X-API-Key": GOOD_KEY})
        r = client.get(f"/history/{sid}", headers={"X-API-Key": GOOD_KEY})
        assert r.status_code == 404


# ── Cost Logger ───────────────────────────────────────────────────────────────────

class TestCostLogger:
    def test_returns_positive_tokens(self):
        from api.services.cost_logger import CostLogger
        u = CostLogger().estimate("What is X?", "X is Y.", ["sub1", "sub2"])
        assert u.total_tokens > 0
        assert u.prompt_tokens > 0
        assert u.completion_tokens > 0

    def test_cost_positive(self):
        from api.services.cost_logger import CostLogger
        u = CostLogger().estimate("q", "a " * 200, ["sub1", "sub2", "sub3"])
        assert u.estimated_cost_usd > 0