"""
agent/decomposer.py
Asks the LLM to break a research question into 3–5 focused sub-questions.
Each sub-question will become one search query in the ReAct loop.
"""
from __future__ import annotations
import json
from agent.client import client, OPENAI_MODEL

_SYSTEM = """\
You are a research planning assistant.
Given a research question, break it into 3 to 5 focused sub-questions
that together would fully answer the original question.

Rules:
- Each sub-question must be specific and searchable.
- Avoid overlap between sub-questions.
- Return ONLY a JSON array of strings, no other text.

Example output:
["What is X?", "How does X work?", "What are the limitations of X?"]
"""


def decompose(question: str) -> list[str]:
    """
    Returns a list of 3–5 sub-questions derived from the main question.
    Falls back to [question] if the LLM returns unparseable output.
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": f"Research question: {question}"},
        ],
        max_tokens=300,
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()

    try:
        sub_questions = json.loads(raw)
        if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions):
            return sub_questions[:5]   # cap at 5
    except json.JSONDecodeError:
        pass

    # Fallback — treat the whole question as one search
    return [question]