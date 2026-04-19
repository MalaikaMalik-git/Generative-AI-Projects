"""
rag/memory.py
─────────────
Multi-turn conversation memory for the RAG tutor.

Responsibilities
────────────────
• Store the full turn history (user + assistant).
• Detect when a new question repeats a prior one (semantic duplicate).
• Summarize the history when it grows beyond a configurable token budget.
• Expose a clean conversation context string for the generator.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class Turn:
    role: str           # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    route: str = "rag"  # "rag" | "tool"
    sources: List[str] = field(default_factory=list)


@dataclass
class ConversationMemory:
    """
    Holds the complete conversation and exposes helpers for
    context injection, duplicate detection, and summarisation.
    """
    max_turns_before_summary: int = 10   # summarise after this many full turns
    summary_keep_last_n: int = 4         # keep this many recent turns verbatim after summary
    similarity_threshold: float = 0.82   # cosine sim above which we flag a repeat

    # internal state
    turns: List[Turn] = field(default_factory=list)
    summary: str = ""                    # rolling compressed summary of older turns
    _embedder = None                     # lazy-loaded sentence-transformer

    # ── public helpers ────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        self.turns.append(Turn(role="user", content=content))

    def add_assistant(
        self,
        content: str,
        route: str = "rag",
        sources: Optional[List[str]] = None,
    ) -> None:
        self.turns.append(
            Turn(
                role="assistant",
                content=content,
                route=route,
                sources=sources or [],
            )
        )

    @property
    def turn_count(self) -> int:
        """Number of complete (user + assistant) pairs."""
        return len([t for t in self.turns if t.role == "user"])

    @property
    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def all_user_questions(self) -> List[str]:
        return [t.content for t in self.turns if t.role == "user"]

    # ── duplicate detection ───────────────────────────────────────

    def find_duplicate(self, query: str) -> Optional[str]:
        """
        Return the previous answer if this query is semantically near-identical
        to a prior user question, else None.
        """
        prior_questions = self.all_user_questions()
        if not prior_questions:
            return None

        try:
            embedder = self._get_embedder()
            if embedder is None:
                return self._lexical_duplicate(query, prior_questions)

            import numpy as np
            all_texts = prior_questions + [query]
            vecs = embedder.encode(all_texts, normalize_embeddings=True)
            query_vec = vecs[-1]
            prior_vecs = vecs[:-1]

            sims = prior_vecs @ query_vec  # cosine similarity (normalized)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= self.similarity_threshold:
                # find the assistant reply that followed the matched question
                return self._answer_for_question_index(best_idx)
        except Exception:
            return self._lexical_duplicate(query, prior_questions)

        return None

    def _lexical_duplicate(self, query: str, prior_questions: List[str]) -> Optional[str]:
        """Fallback: token-overlap Jaccard similarity."""
        def tokens(s: str):
            return set(re.findall(r"\w+", s.lower()))

        q_tokens = tokens(query)
        for idx, pq in enumerate(prior_questions):
            pq_tokens = tokens(pq)
            union = q_tokens | pq_tokens
            if not union:
                continue
            jaccard = len(q_tokens & pq_tokens) / len(union)
            if jaccard >= 0.80:
                return self._answer_for_question_index(idx)
        return None

    def _answer_for_question_index(self, question_idx: int) -> Optional[str]:
        """Return the assistant reply paired with the n-th user question."""
        user_turn_count = 0
        for i, turn in enumerate(self.turns):
            if turn.role == "user":
                if user_turn_count == question_idx:
                    # look for next assistant turn
                    for j in range(i + 1, len(self.turns)):
                        if self.turns[j].role == "assistant":
                            return self.turns[j].content
                user_turn_count += 1
        return None

    # ── context building ──────────────────────────────────────────

    def build_context_string(self) -> str:
        """
        Returns a compact conversation-history string to inject into the
        generator prompt.  Includes the rolling summary (if any) and the
        most recent turns.
        """
        parts: List[str] = []

        if self.summary:
            parts.append(f"[Conversation summary so far]\n{self.summary}")

        # Always include at least the last `summary_keep_last_n` pairs
        recent = self._recent_turns(self.summary_keep_last_n * 2)
        if recent:
            parts.append("[Recent conversation]")
            for turn in recent:
                speaker = "Student" if turn.role == "user" else "Tutor"
                parts.append(f"{speaker}: {turn.content}")

        return "\n\n".join(parts)

    def _recent_turns(self, n: int) -> List[Turn]:
        return self.turns[-n:] if len(self.turns) > n else self.turns[:]

    # ── auto-summarisation ────────────────────────────────────────

    def maybe_summarise(self) -> bool:
        """
        If the conversation exceeds `max_turns_before_summary`, compress
        older turns into a rolling summary.  Returns True if summarised.
        """
        if self.turn_count < self.max_turns_before_summary:
            return False

        keep = self.summary_keep_last_n * 2          # keep N pairs verbatim
        old_turns = self.turns[:-keep] if len(self.turns) > keep else []
        if not old_turns:
            return False

        new_summary = self._compress_turns(old_turns)
        if self.summary:
            self.summary = f"{self.summary}\n\n{new_summary}"
        else:
            self.summary = new_summary

        self.turns = self.turns[-keep:]
        return True

    def _compress_turns(self, turns: List[Turn]) -> str:
        """
        Compress a list of turns into a bullet-point summary.
        Uses OpenAI if available, otherwise creates a structured digest.
        """
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)

                transcript = "\n".join(
                    f"{'Student' if t.role == 'user' else 'Tutor'}: {t.content}"
                    for t in turns
                )
                prompt = (
                    "Summarise the following tutoring conversation into "
                    "3-7 concise bullet points. Capture the key questions "
                    "asked, the main concepts explained, and any conclusions "
                    "reached. Be factual and brief.\n\n"
                    f"{transcript}"
                )
                response = client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                )
                return response.output_text.strip()
            except Exception:
                pass

        # Fallback: structured digest without LLM
        lines = ["Topics covered in earlier conversation:"]
        user_turns = [t for t in turns if t.role == "user"]
        for i, t in enumerate(user_turns[:8], 1):
            lines.append(f"  {i}. {t.content[:120]}")
        return "\n".join(lines)

    # ── topic tracking ────────────────────────────────────────────

    def topic_summary(self) -> List[str]:
        """Return a short list of the distinct topics asked so far."""
        questions = self.all_user_questions()
        # Simple keyword extraction
        stop = {"what", "why", "how", "is", "are", "the", "a", "an", "does",
                "do", "when", "where", "can", "would", "should", "and", "or",
                "but", "in", "on", "for", "of", "to", "it", "its", "that",
                "this", "with", "not", "be"}
        seen: set[str] = set()
        topics: List[str] = []
        for q in questions[-12:]:
            words = [w for w in re.findall(r"\w+", q.lower()) if w not in stop and len(w) > 3]
            key = " ".join(words[:3])
            if key and key not in seen:
                seen.add(key)
                topics.append(q[:60] + ("…" if len(q) > 60 else ""))
        return topics

    # ── private ───────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self._embedder = None
        return self._embedder