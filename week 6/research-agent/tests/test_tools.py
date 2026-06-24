"""
tests/test_tools.py
Unit tests for tools/search.py and tools/fetch.py.
All network calls are mocked — no real HTTP requests made.
"""
from __future__ import annotations
import json
import pytest
from unittest.mock import MagicMock, patch


# ── Search tool tests ──────────────────────────────────────────────────────────

class TestWebSearch:

    @patch("tools.search.DDGS")
    def test_returns_json_list(self, mock_ddgs):
        """Should return a JSON array of results."""
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Content 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Content 2"},
        ]
        from tools.search import web_search
        result = web_search("test query")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["title"] == "Result 1"
        assert parsed[0]["url"] == "https://example.com/1"

    @patch("tools.search.DDGS")
    def test_normalises_href_to_url(self, mock_ddgs):
        """ddgs returns 'href' — we should rename it to 'url'."""
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "T", "href": "https://example.com", "body": "B"},
        ]
        from tools.search import web_search
        result = json.loads(web_search("query"))
        assert "url" in result[0]
        assert "href" not in result[0]

    @patch("tools.search.DDGS")
    def test_empty_results_returns_no_results_message(self, mock_ddgs):
        """Empty DDG results should return a helpful message, not crash."""
        mock_ddgs.return_value.__enter__.return_value.text.return_value = []
        # Second call (retry) also empty
        from tools.search import web_search
        with patch("tools.search.time.sleep"):
            result = json.loads(web_search("obscure query"))
        assert result[0]["title"] == "No results found"

    @patch("tools.search.DDGS")
    def test_exception_returns_error_message(self, mock_ddgs):
        """If DDG raises, return an error string — don't propagate exception."""
        mock_ddgs.return_value.__enter__.return_value.text.side_effect = Exception("network error")
        from tools.search import web_search
        result = json.loads(web_search("query"))
        assert "error" in result[0]["title"].lower()


# ── Fetch tool tests ───────────────────────────────────────────────────────────

class TestFetchPage:

    @patch("tools.fetch.requests.get")
    def test_returns_plain_text(self, mock_get):
        """Should strip HTML and return clean text."""
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><article><p>Hello world</p></article></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from tools.fetch import fetch_page
        result = fetch_page("https://example.com")
        assert "Hello world" in result
        assert "<" not in result   # no HTML tags

    @patch("tools.fetch.requests.get")
    def test_strips_script_and_style(self, mock_get):
        """Script and style content should be removed."""
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><script>alert('x')</script><p>Real content</p></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from tools.fetch import fetch_page
        result = fetch_page("https://example.com")
        assert "alert" not in result
        assert "Real content" in result

    @patch("tools.fetch.requests.get")
    def test_truncates_long_pages(self, mock_get):
        """Pages longer than 3000 chars should be truncated."""
        mock_resp = MagicMock()
        mock_resp.text = f"<body><p>{'x' * 5000}</p></body>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from tools.fetch import fetch_page
        result = fetch_page("https://example.com")
        assert len(result) <= 3100   # some slack for the "[truncated]" suffix
        assert "truncated" in result

    def test_invalid_url_returns_error(self):
        """Non-HTTP URLs should return an error string immediately."""
        from tools.fetch import fetch_page
        result = fetch_page("not-a-url")
        assert "Error" in result

    @patch("tools.fetch.requests.get")
    def test_timeout_returns_error(self, mock_get):
        """Timeouts should return a human-readable error, not raise."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        from tools.fetch import fetch_page
        result = fetch_page("https://example.com")
        assert "timed out" in result

    @patch("tools.fetch.requests.get")
    def test_http_error_returns_error(self, mock_get):
        """HTTP 404/403 should return an error string."""
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )

        from tools.fetch import fetch_page
        result = fetch_page("https://example.com/missing")
        assert "Error" in result