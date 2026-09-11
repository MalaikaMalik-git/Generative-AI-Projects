"""
LLM helper used by the /chat endpoint.

Uses OpenAI (gpt-4o-mini) so the same provider and API key can be reused
for vision in vision.py, per the project plan's "one AI provider for both
answers and vision when possible."
"""

import os

from openai import OpenAI

MODEL = "gpt-4o-mini"
MAX_TOKENS = 400

SYSTEM_PROMPT = """You are the voice of a 3D avatar assistant embedded on a company website. \
Visitors ask you questions about the company, and you answer using ONLY the \
context provided below each question — never your general knowledge.

Rules:
- If the answer is in the context, answer confidently and naturally, as if you \
just know it — don't say "according to the context" or "based on the provided text."
- If the answer is NOT in the context, say plainly that you don't have that \
information, and suggest what you can help with instead. Never invent facts.
- Keep answers short: 2-4 sentences. Your answer will be read aloud by \
text-to-speech, so avoid bullet points, markdown, or long lists.
- Speak like a helpful company representative, not a search engine."""

_client = None  # lazy singleton


class LLMNotConfiguredError(RuntimeError):
    """Raised when OPENAI_API_KEY is missing."""


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMNotConfiguredError(
            "OPENAI_API_KEY is not set. Add it to backend/.env "
            "(get a key at https://platform.openai.com/api-keys)."
        )

    _client = OpenAI(api_key=api_key)
    return _client


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    client = get_client()

    context_text = _format_context(context_chunks)
    user_message = (
        f"Context from the company website:\n\n{context_text}\n\n"
        f"Visitor question: {question}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content.strip()
