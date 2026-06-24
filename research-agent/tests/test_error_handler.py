"""
tests/test_error_handler.py
Tests for all 4 failure cases — fully mocked, no real network calls.

Failure cases:
  1. Search returns no results       → retries with rephrased query
  2. Search API error (protocol/SSL) → retries with delay, then fallback
  3. Max steps exceeded              → partial answer returned gracefully
  4. Synthesizer fails               → fallback report returned
"""
from __future__ import annotations
import json
import pytest
from unittest.mock import patch, MagicMock
from agent.models import Step, StepType


# ── Failure Case 1 & 2: safe_search ───────────────────────────────────────────

class TestSafeSearch:

    @patch("agent.error_handler.time.sleep")
    @patch("tools.search.DDGS")
    def test_retries_on_no_results(self, mock_ddgs, mock_sleep):
        """Should retry when search returns empty, then succeed on second try."""
        good_result = [{"title": "Real Result", "href": "https://example.com", "body": "Real content"}]
        mock_ddgs.return_value.__enter__.return_value.text.side_effect = [
            [],           # first call: empty
            good_result,  # second call: success
        ]
        from agent.error_handler import safe_search
        result = json.loads(safe_search("test query", retries=2, delay=0))
        assert result[0]["title"] == "Real Result"

    @patch("agent.error_handler.time.sleep")
    @patch("tools.search.DDGS")
    def test_returns_error_sentinel_after_all_retries(self, mock_ddgs, mock_sleep):
        """Should return 'Search unavailable' sentinel after all retries exhausted."""
        mock_ddgs.return_value.__enter__.return_value.text.return_value = []
        from agent.error_handler import safe_search
        result = json.loads(safe_search("bad query", retries=2, delay=0))
        assert result[0]["title"] == "Search unavailable"
        assert "Could not retrieve" in result[0]["body"]

    @patch("agent.error_handler.time.sleep")
    @patch("tools.search.DDGS")
    def test_retries_on_exception(self, mock_ddgs, mock_sleep):
        """Should retry on exception and return fallback if all fail."""
        mock_ddgs.return_value.__enter__.return_value.text.side_effect = Exception("SSL error")
        from agent.error_handler import safe_search
        result = json.loads(safe_search("query", retries=2, delay=0))
        assert result[0]["title"] == "Search unavailable"

    @patch("agent.error_handler.time.sleep")
    @patch("tools.search.DDGS")
    def test_success_on_first_try(self, mock_ddgs, mock_sleep):
        """Should return results immediately if first search succeeds."""
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Good", "href": "https://good.com", "body": "Good content"}
        ]
        from agent.error_handler import safe_search
        result = json.loads(safe_search("good query", retries=2, delay=0))
        assert result[0]["title"] == "Good"
        mock_sleep.assert_not_called()   # no delay needed


# ── Failure Case 3: handle_max_steps ──────────────────────────────────────────

class TestHandleMaxSteps:

    def test_returns_partial_answer_from_observations(self):
        """Should extract observations and return a partial answer string."""
        steps = [
            Step(StepType.THOUGHT, "thinking..."),
            Step(StepType.OBSERVATION, "JWST has a 6.5m mirror"),
            Step(StepType.OBSERVATION, "It was launched in 2021"),
        ]
        from agent.error_handler import handle_max_steps
        result = handle_max_steps("What is JWST?", steps)
        assert "JWST" in result or "6.5m" in result
        assert isinstance(result, str)

    def test_handles_no_observations_gracefully(self):
        """Should return a helpful message if there are no observations at all."""
        from agent.error_handler import handle_max_steps
        result = handle_max_steps("What is JWST?", [])
        assert "unable to complete" in result.lower() or "no search" in result.lower()


# ── Failure Case 4: safe_synthesize ───────────────────────────────────────────

class TestSafeSynthesize:

    @patch("agent.synthesizer.client")
    def test_returns_normal_report_on_success(self, mock_client):
        """Should return synthesized report when no error occurs."""
        msg = MagicMock(); msg.content = "## Summary\nJWST is great."
        choice = MagicMock(); choice.message = msg
        resp = MagicMock(); resp.choices = [choice]
        mock_client.chat.completions.create.return_value = resp

        from agent.error_handler import safe_synthesize
        result = safe_synthesize("What is JWST?", [
            {"question": "What is JWST?", "answer": "A telescope.", "success": True}
        ])
        assert "Summary" in result

    @patch("agent.synthesizer.client")
    def test_returns_fallback_on_synthesizer_failure(self, mock_client):
        """Should return fallback report when synthesizer raises an exception."""
        mock_client.chat.completions.create.side_effect = Exception("API timeout")

        from agent.error_handler import safe_synthesize
        result = safe_synthesize("What is JWST?", [
            {"question": "What is JWST?", "answer": "A telescope.", "success": True}
        ])
        # Fallback report should still contain the answer
        assert "telescope" in result.lower()
        assert isinstance(result, str)