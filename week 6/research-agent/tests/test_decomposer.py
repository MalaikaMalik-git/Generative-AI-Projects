"""
tests/test_decomposer.py
Unit tests for the decomposer — LLM call is mocked.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from agent.decomposer import decompose


def _mock_response(content: str):
    msg    = MagicMock(); msg.content = content
    choice = MagicMock(); choice.message = msg
    resp   = MagicMock(); resp.choices = [choice]
    return resp


@patch("agent.decomposer.client")
def test_decompose_returns_list(mock_client):
    """Should return a list of strings."""
    mock_client.chat.completions.create.return_value = _mock_response(
        json.dumps(["What is X?", "How does X work?", "Why is X important?"])
    )
    result = decompose("Tell me about X")
    assert isinstance(result, list)
    assert all(isinstance(q, str) for q in result)


@patch("agent.decomposer.client")
def test_decompose_caps_at_five(mock_client):
    """Should return at most 5 sub-questions."""
    mock_client.chat.completions.create.return_value = _mock_response(
        json.dumps([f"Question {i}?" for i in range(10)])
    )
    result = decompose("Big question")
    assert len(result) <= 5


@patch("agent.decomposer.client")
def test_decompose_fallback_on_bad_json(mock_client):
    """Should fall back to [original_question] if LLM returns bad JSON."""
    mock_client.chat.completions.create.return_value = _mock_response(
        "Sorry, I cannot decompose this."
    )
    result = decompose("My question")
    assert result == ["My question"]