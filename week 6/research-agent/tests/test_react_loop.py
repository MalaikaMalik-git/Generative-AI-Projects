"""
tests/test_react_loop.py
Unit tests for the ReAct loop — all LLM calls are mocked so these run
instantly without spending any API credits.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from agent.models    import StepType, AgentResult
from agent.react_loop import run


def _make_tool_response(fn_name: str, fn_args: dict, tool_call_id: str = "tc_001"):
    """Helper: build a mock OpenAI response that calls a tool."""
    tool_call = MagicMock()
    tool_call.id = tool_call_id
    tool_call.function.name = fn_name
    tool_call.function.arguments = json.dumps(fn_args)

    msg = MagicMock()
    msg.content = f"I need to search for {fn_args}"
    msg.tool_calls = [tool_call]

    choice = MagicMock()
    choice.message = msg

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_text_response(text: str):
    """Helper: build a mock OpenAI response with plain text (Final Answer)."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None

    choice = MagicMock()
    choice.message = msg

    response = MagicMock()
    response.choices = [choice]
    return response


# ── Tests ──────────────────────────────────────────────────────────────────────

@patch("agent.react_loop.client")
def test_final_answer_on_first_text(mock_client):
    """Agent should stop immediately when LLM returns a Final Answer."""
    mock_client.chat.completions.create.return_value = _make_text_response(
        "Final Answer: Python is a high-level programming language."
    )

    result = run("What is Python?", verbose=False)

    assert isinstance(result, AgentResult)
    assert result.success is True
    assert "Python" in result.answer
    assert result.error is None


@patch("agent.react_loop.client")
def test_tool_call_then_answer(mock_client):
    """Agent should call a tool, receive observation, then return Final Answer."""
    mock_client.chat.completions.create.side_effect = [
        _make_tool_response("web_search", {"query": "Python programming language"}),
        _make_text_response("Final Answer: Python is a versatile language used in AI."),
    ]

    result = run("What is Python?", verbose=False)

    assert result.success is True
    # Trace should contain Thought, Action, Observation, then Answer
    step_types = [s.type for s in result.steps]
    assert StepType.ACTION      in step_types
    assert StepType.OBSERVATION in step_types
    assert StepType.ANSWER      in step_types


@patch("agent.react_loop.client")
def test_max_steps_exceeded(mock_client):
    """Agent should stop gracefully and set error=MAX_STEPS_EXCEEDED."""
    # Always return a tool call — loop will never find Final Answer
    mock_client.chat.completions.create.return_value = _make_tool_response(
        "web_search", {"query": "infinite loop test"}
    )

    with patch("agent.react_loop.MAX_STEPS", 3):
        result = run("Loop forever?", verbose=False)

    assert result.success is False
    assert result.error == "MAX_STEPS_EXCEEDED"


@patch("agent.react_loop.client")
def test_unknown_tool_is_handled(mock_client):
    """Unknown tool calls should return an error observation, not crash."""
    mock_client.chat.completions.create.side_effect = [
        _make_tool_response("nonexistent_tool", {"x": 1}),
        _make_text_response("Final Answer: Could not complete research."),
    ]

    result = run("Trigger unknown tool", verbose=False)
    # Should not raise — error observation is fed back and loop continues
    assert result is not None


@patch("agent.react_loop.client")
def test_step_trace_ordering(mock_client):
    """Trace steps must follow Thought → Action → Observation order."""
    mock_client.chat.completions.create.side_effect = [
        _make_tool_response("web_search", {"query": "test"}),
        _make_text_response("Final Answer: Done."),
    ]

    result = run("Test ordering", verbose=False)
    types = [s.type for s in result.steps]

    # First action must come after first thought
    thought_idx = types.index(StepType.THOUGHT)
    action_idx  = types.index(StepType.ACTION)
    obs_idx     = types.index(StepType.OBSERVATION)

    assert thought_idx < action_idx < obs_idx