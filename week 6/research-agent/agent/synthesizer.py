"""
agent/synthesizer.py
Takes all sub-question answers and synthesizes them into one
structured, cited final report.
"""
from __future__ import annotations
import re
from agent.client import client, OPENAI_MODEL

_SYSTEM = """\
You are a research report writer. You will receive a main research question
and a set of sub-question answers with their sources.

Write a structured final report with these exact sections:

## Summary
2-3 sentence overview answering the main question directly.

## Key Findings
3-5 bullet points of the most important facts discovered.

## Detailed Analysis
2-3 short paragraphs expanding on the findings.

## Sources
Numbered list of all unique URLs cited, one per line, like:
1. https://...
2. https://...

Rules:
- Be factual and concise.
- Every claim must come from the provided research.
- Do not invent information not present in the sub-answers.
- Extract all URLs from the sub-answers and list them in Sources.
"""


def _extract_urls(text: str) -> list[str]:
    """Pull all URLs out of a block of text."""
    return re.findall(r'https?://[^\s\)\]>,;"\']+', text)


def synthesize(question: str, sub_results: list[dict]) -> str:
    """
    Synthesize sub-question results into a final structured report.

    Args:
        question:    The original top-level research question.
        sub_results: List of {"question": str, "answer": str} dicts.

    Returns:
        A formatted markdown report string.
    """
    # Build the context block fed to the LLM
    research_block = ""
    for i, r in enumerate(sub_results, 1):
        research_block += f"\n### Sub-question {i}: {r['question']}\n"
        research_block += f"{r['answer']}\n"

    prompt = f"""\
Main research question: {question}

Research findings:
{research_block}

Write the structured report now.
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()