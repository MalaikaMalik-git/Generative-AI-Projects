"""
tests/test_complex_query.py
Tests that verify the agent handles complex, multi-faceted questions correctly.
All LLM and search calls are mocked — these run instantly with no API cost.

A "complex query" is one that:
  - Requires multiple sub-questions (3+)
  - Involves comparison, causation, or multi-domain reasoning
  - Produces a structured report with multiple sections
"""
from __future__ import annotations
import json
import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_llm_text(text: str):
    msg    = MagicMock(); msg.content = text; msg.tool_calls = None
    choice = MagicMock(); choice.message = msg
    resp   = MagicMock(); resp.choices = [choice]
    return resp


def _mock_llm_tool_call(fn_name: str, fn_args: dict, tool_call_id: str = "tc_001"):
    tool_call = MagicMock()
    tool_call.id = tool_call_id
    tool_call.function.name = fn_name
    tool_call.function.arguments = json.dumps(fn_args)

    msg = MagicMock()
    msg.content = f"Searching for {fn_args}"
    msg.tool_calls = [tool_call]

    choice = MagicMock(); choice.message = msg
    resp   = MagicMock(); resp.choices = [choice]
    return resp


FAKE_SEARCH_RESULT = json.dumps([
    {"title": "Result 1", "url": "https://example.com/1", "body": "Relevant content about the topic with key facts and data."},
    {"title": "Result 2", "url": "https://example.com/2", "body": "More relevant content providing additional context and details."},
])


# ── Test: Decomposer handles complex queries ───────────────────────────────────

class TestComplexDecomposition:

    @patch("agent.decomposer.client")
    def test_complex_query_produces_multiple_subquestions(self, mock_client):
        """A complex multi-faceted question should decompose into 3+ sub-questions."""
        sub_qs = [
            "What caused the 2008 financial crisis?",
            "Which banks were most affected by the 2008 crisis?",
            "What regulations were introduced after 2008?",
            "How did the crisis affect global economies?",
        ]
        msg    = MagicMock(); msg.content = json.dumps(sub_qs)
        choice = MagicMock(); choice.message = msg
        resp   = MagicMock(); resp.choices = [choice]
        mock_client.chat.completions.create.return_value = resp

        from agent.decomposer import decompose
        result = decompose(
            "What caused the 2008 financial crisis and what were its global consequences?"
        )
        assert len(result) >= 3
        assert all(isinstance(q, str) for q in result)

    @patch("agent.decomposer.client")
    def test_comparison_query_decomposition(self, mock_client):
        """Comparison questions should be broken into aspects of each side."""
        sub_qs = [
            "What are the key features of Python?",
            "What are the key features of JavaScript?",
            "How do Python and JavaScript differ in performance?",
        ]
        msg    = MagicMock(); msg.content = json.dumps(sub_qs)
        choice = MagicMock(); choice.message = msg
        resp   = MagicMock(); resp.choices = [choice]
        mock_client.chat.completions.create.return_value = resp

        from agent.decomposer import decompose
        result = decompose("Compare Python and JavaScript for web development")
        assert len(result) >= 2


# ── Test: ReAct loop handles multi-step reasoning ─────────────────────────────

class TestMultiStepReasoning:

    @patch("agent.react_loop.client")
    @patch("agent.error_handler.time.sleep")
    @patch("tools.search.DDGS")
    def test_multi_step_search_then_answer(self, mock_ddgs, mock_sleep, mock_llm):
        """Agent should search, observe results, then produce Final Answer."""
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "T", "href": "https://example.com", "body": "Relevant content"}
        ]
        mock_llm.chat.completions.create.side_effect = [
            _mock_llm_tool_call("web_search", {"query": "complex topic"}),
            _mock_llm_text("Final Answer: This is a comprehensive answer with multiple facts cited from https://example.com."),
        ]

        from agent.react_loop import run
        result = run("What is a complex multi-faceted topic?", verbose=False)

        assert result.success is True
        assert len(result.steps) >= 3   # Thought + Action + Observation + Answer
        assert "Final Answer" not in result.answer   # stripped correctly

    @patch("agent.react_loop.client")
    @patch("agent.error_handler.time.sleep")
    @patch("tools.search.DDGS")
    def test_agent_recovers_from_bad_first_search(self, mock_ddgs, mock_sleep, mock_llm):
        """Agent should still produce an answer even if first search fails."""
        # First search: empty (triggers retry in safe_search)
        # Second call returns good results
        mock_ddgs.return_value.__enter__.return_value.text.side_effect = [
            [],  # first search: empty
            [{"title": "Good Result", "href": "https://example.com", "body": "Good content"}],
        ]
        mock_llm.chat.completions.create.side_effect = [
            _mock_llm_tool_call("web_search", {"query": "complex topic"}),
            _mock_llm_text("Final Answer: Found the answer after retry."),
        ]

        from agent.react_loop import run
        result = run("Complex question requiring retry", verbose=False)
        assert result is not None   # didn't crash


# ── Test: Full pipeline on complex query ──────────────────────────────────────

class TestFullPipelineComplexQuery:

    @patch("agent.synthesizer.client")
    @patch("agent.react_loop.client")
    @patch("agent.decomposer.client")
    @patch("agent.error_handler.time.sleep")
    @patch("tools.search.DDGS")
    def test_full_pipeline_complex_question(
        self, mock_ddgs, mock_sleep, mock_decomp, mock_react, mock_synth
    ):
        """End-to-end test: decompose → search → synthesize on a complex question."""

        # Decomposer returns 3 sub-questions
        sub_qs = ["Sub Q1?", "Sub Q2?", "Sub Q3?"]
        decomp_msg = MagicMock(); decomp_msg.content = json.dumps(sub_qs)
        decomp_choice = MagicMock(); decomp_choice.message = decomp_msg
        decomp_resp = MagicMock(); decomp_resp.choices = [decomp_choice]
        mock_decomp.chat.completions.create.return_value = decomp_resp

        # Search returns real-looking results
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Result", "href": "https://example.com", "body": "Relevant content."}
        ]

        # ReAct loop: search then answer (for each of 3 sub-questions)
        react_responses = []
        for i in range(3):
            react_responses.append(_mock_llm_tool_call("web_search", {"query": f"sub q {i}"}))
            react_responses.append(_mock_llm_text(f"Final Answer: Answer to sub question {i+1} (source: https://example.com/{i})."))
        mock_react.chat.completions.create.side_effect = react_responses

        # Synthesizer returns a structured report
        synth_msg = MagicMock()
        synth_msg.content = "## Summary\nComprehensive answer.\n\n## Key Findings\n- Fact 1\n- Fact 2\n\n## Sources\n1. https://example.com"
        synth_choice = MagicMock(); synth_choice.message = synth_msg
        synth_resp = MagicMock(); synth_resp.choices = [synth_choice]
        mock_synth.chat.completions.create.return_value = synth_resp

        from agent.decomposer     import decompose
        from agent.react_loop     import run
        from agent.error_handler  import safe_synthesize

        question = "Compare the economic impacts of AI on developed vs developing countries"
        sub_questions = decompose(question)
        assert len(sub_questions) == 3

        sub_results = []
        for sub_q in sub_questions:
            result = run(sub_q, verbose=False)
            sub_results.append({
                "question": sub_q,
                "answer":   result.answer,
                "success":  result.success,
            })

        assert all(r["success"] for r in sub_results)

        report = safe_synthesize(question, sub_results)
        assert "Summary" in report
        assert isinstance(report, str)
        assert len(report) > 50