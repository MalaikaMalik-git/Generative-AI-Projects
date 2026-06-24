"""
agent/error_handler.py
Handles all failure cases gracefully so the agent never crashes.

Failure cases covered (all demo-ready):
  1. Search returns no results       → rephrase and retry once
  2. Search API error (protocol/SSL) → retry with delay, then fallback
  3. Max steps exceeded              → return partial answer gracefully
  4. Synthesizer fails               → return raw sub-answers as fallback
"""
from __future__ import annotations
import time
import json
from rich.console import Console

console = Console()


class AgentError(Exception):
    """Base class for agent errors."""
    pass


class SearchFailedError(AgentError):
    pass


class MaxStepsError(AgentError):
    pass


class SynthesisError(AgentError):
    pass


def safe_search(query: str, retries: int = 2, delay: float = 2.0) -> str:
    """
    Run web_search with automatic retry on failure.

    Failure case 1: Empty results → rephrased retry
    Failure case 2: Protocol/network error → wait and retry

    Returns JSON string of results, or an error-result JSON on total failure.
    """
    from tools.search import web_search

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            result = web_search(query)
            parsed = json.loads(result)

            # Check for error sentinel returned by web_search
            if parsed and parsed[0].get("title") in ("Search error", "No results found"):
                last_error = parsed[0].get("body", "Unknown search error")
                console.print(
                    f"[yellow]⚠ Search attempt {attempt} failed: {last_error[:80]}[/yellow]"
                )
                if attempt < retries:
                    # Rephrase by stripping the last word (simplify query)
                    words = query.split()
                    query = " ".join(words[:-1]) if len(words) > 2 else query
                    console.print(f"[dim]  Retrying with: '{query}'[/dim]")
                    time.sleep(delay)
                continue

            return result  # success

        except Exception as e:
            last_error = str(e)
            console.print(f"[yellow]⚠ Search attempt {attempt} raised: {last_error[:80]}[/yellow]")
            if attempt < retries:
                time.sleep(delay)

    # All retries exhausted — return a graceful error result
    console.print("[red]✗ Search failed after all retries — returning empty result.[/red]")
    return json.dumps([{
        "title": "Search unavailable",
        "url":   "",
        "body":  (
            f"Could not retrieve search results for '{query}' after {retries} attempts. "
            f"Last error: {last_error}. "
            "The agent will attempt to answer from available context."
        ),
    }])


def safe_synthesize(question: str, sub_results: list[dict]) -> str:
    """
    Run synthesizer with a fallback to raw answers if it fails.

    Failure case 4: Synthesizer API call fails → format raw answers manually.
    """
    from agent.synthesizer import synthesize

    try:
        return synthesize(question, sub_results)
    except Exception as e:
        console.print(f"[yellow]⚠ Synthesizer failed: {e}. Using fallback format.[/yellow]")
        return _fallback_report(question, sub_results)


def handle_max_steps(question: str, steps: list) -> str:
    """
    Failure case 3: Agent hit MAX_STEPS without a Final Answer.
    Extract all observations and build a partial answer.
    """
    from agent.models import StepType
    console.print(
        "[yellow]⚠ Max steps reached — generating partial answer from observations.[/yellow]"
    )

    observations = [s.content for s in steps if s.type == StepType.OBSERVATION]

    if not observations:
        return (
            f"The agent was unable to complete research on: '{question}'. "
            "No search results were retrieved. Please try a simpler or different question."
        )

    combined = " ".join(observations)[:800]
    return (
        f"[Partial answer — max research steps reached]\n\n"
        f"Based on available search results for '{question}':\n\n"
        f"{combined}..."
    )


def _fallback_report(question: str, sub_results: list[dict]) -> str:
    """Format sub-answers manually when synthesizer is unavailable."""
    lines = [f"## Research Results: {question}\n"]
    for r in sub_results:
        status = "✓" if r.get("success") else "⚠ partial"
        lines.append(f"### {r['question']} [{status}]")
        lines.append(r["answer"])
        lines.append("")

    lines.append("## Note")
    lines.append(
        "This report was generated using the fallback formatter "
        "because the synthesis step encountered an error."
    )
    return "\n".join(lines)