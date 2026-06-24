"""
tests/test_synthesizer.py
Unit tests for the synthesizer — LLM call is mocked.
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch


def _mock_response(content: str):
    msg    = MagicMock(); msg.content = content
    choice = MagicMock(); choice.message = msg
    resp   = MagicMock(); resp.choices = [choice]
    return resp


SAMPLE_REPORT = """\
## Summary
The JWST is a powerful infrared telescope launched in 2021.

## Key Findings
- 6.5 meter mirror
- Infrared optimized
- Operates at L2

## Detailed Analysis
The telescope has revolutionized astronomy.

## Sources
1. https://en.wikipedia.org/wiki/James_Webb_Space_Telescope
2. https://science.nasa.gov/mission/webb/
"""

SAMPLE_RESULTS = [
    {"question": "What is JWST?",       "answer": "JWST is an infrared telescope. Source: https://en.wikipedia.org/wiki/James_Webb_Space_Telescope", "success": True},
    {"question": "What has JWST found?", "answer": "It found early galaxies. Source: https://science.nasa.gov/mission/webb/", "success": True},
]


@patch("agent.synthesizer.client")
def test_synthesize_returns_string(mock_client):
    """synthesize() should return a non-empty string."""
    mock_client.chat.completions.create.return_value = _mock_response(SAMPLE_REPORT)
    from agent.synthesizer import synthesize
    result = synthesize("What is JWST?", SAMPLE_RESULTS)
    assert isinstance(result, str)
    assert len(result) > 0


@patch("agent.synthesizer.client")
def test_synthesize_contains_sections(mock_client):
    """Report should contain the main section headers."""
    mock_client.chat.completions.create.return_value = _mock_response(SAMPLE_REPORT)
    from agent.synthesizer import synthesize
    result = synthesize("What is JWST?", SAMPLE_RESULTS)
    assert "Summary"  in result
    assert "Sources"  in result


@patch("agent.synthesizer.client")
def test_synthesize_handles_empty_results(mock_client):
    """Should not crash with empty sub_results list."""
    mock_client.chat.completions.create.return_value = _mock_response("## Summary\nNo data.")
    from agent.synthesizer import synthesize
    result = synthesize("Any question?", [])
    assert isinstance(result, str)


def test_extract_urls():
    """URL extractor should find all URLs in text."""
    from agent.synthesizer import _extract_urls
    text = "See https://nasa.gov and https://wikipedia.org for more."
    urls = _extract_urls(text)
    assert "https://nasa.gov" in urls
    assert "https://wikipedia.org" in urls
    assert len(urls) == 2