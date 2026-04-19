"""
rag/generator.py
────────────────
Answer generator with:
  • Multi-turn conversation context injection
  • Tutor persona and faithfulness rules
  • Graceful retrieval-only fallback
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# Fallback (no API key)
# ──────────────────────────────────────────────

def _fallback_answer(question: str, contexts: List[Dict]) -> str:
    if not contexts:
        return "No relevant context was retrieved for this question."

    snippets = []
    for item in contexts[:3]:
        snippet = item["text"].strip().replace("\n", " ")
        snippets.append(f"[{item['source']}] {snippet[:300]}")

    return (
        "**Retrieval-only mode** — no LLM key is configured.\n\n"
        "Most relevant passages:\n\n"
        + "\n\n".join(snippets)
    )


# ──────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────

TUTOR_SYSTEM_PROMPT = """\
You are an expert AI tutor for a personal knowledge base.

ROLE
────
You help students deeply understand technical concepts from retrieved documents.
You remember the conversation history and build on it naturally.

RULES
─────
1. Answer ONLY from the provided context passages.
2. If the answer is not in the context, say exactly:
   "The provided context does not contain enough information to fully answer this."
   Then offer to rephrase or suggest related topics you have seen earlier.
3. Do NOT hallucinate, guess, or use outside knowledge.
4. If this question is a follow-up to earlier turns, explicitly connect it:
   e.g. "Building on what we discussed about X…"
5. If you already answered a very similar question, acknowledge it and either
   expand or redirect: "We covered this earlier — let me add more detail…"
6. Keep answers focused, structured, and pedagogically clear.
7. Use markdown: headers, bullet points, and bold for key terms.
8. End with a one-sentence "Key takeaway:" when answering knowledge questions.
"""


def generate_answer(
    question: str,
    contexts: List[Dict],
    conversation_context: str = "",
) -> str:
    """
    Generate an answer grounded in `contexts`.

    Parameters
    ----------
    question             : current user question
    contexts             : retrieved RAG chunks (list of dicts with 'text', 'source')
    conversation_context : formatted prior-turn history from ConversationMemory
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

    if not api_key:
        return _fallback_answer(question, contexts)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        context_block = "\n\n".join(
            f"[Source: {item['source']}]\n{item['text']}"
            for item in contexts[:3]
        )

        # Build the user message – include prior conversation if available
        user_parts: List[str] = []

        if conversation_context:
            user_parts.append(f"CONVERSATION HISTORY\n{conversation_context}")

        user_parts.append(f"RETRIEVED CONTEXT\n{context_block}")
        user_parts.append(f"CURRENT QUESTION\n{question}")

        user_message = "\n\n" + "\n\n---\n\n".join(user_parts)

        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        return response.output_text.strip()

    except Exception as exc:
        return (
            f"⚠️ LLM generation failed — showing retrieval-only mode.\n\n"
            f"**Reason:** {exc}\n\n"
            + _fallback_answer(question, contexts)
        )


def generate_standalone_question(
    question: str,
    conversation_context: str,
) -> str:
    """
    Rewrite a follow-up question into a self-contained question
    so the retriever can find better chunks.

    e.g. "What about the overlap part?" →
         "What is chunk overlap in RAG and why is it important?"
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

    if not api_key or not conversation_context:
        return question

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = (
            "Given the conversation history below and the follow-up question, "
            "rewrite the follow-up question as a single, self-contained question "
            "that a search engine could answer without any prior context.\n\n"
            f"CONVERSATION HISTORY:\n{conversation_context}\n\n"
            f"FOLLOW-UP QUESTION: {question}\n\n"
            "REWRITTEN QUESTION (output ONLY the question, nothing else):"
        )

        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
        )

        rewritten = response.output_text.strip()
        # Safety: if the rewrite is too long or weird, fall back
        if 10 < len(rewritten) < 300:
            return rewritten
    except Exception:
        pass

    return question