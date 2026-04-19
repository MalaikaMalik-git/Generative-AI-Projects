"""
rag/pipeline.py
───────────────
RAGPipeline — now multi-turn aware.

Changes vs v1
─────────────
• Accepts an optional ConversationMemory instance.
• Rewrites follow-up questions before retrieval (query contextualisation).
• Detects near-duplicate questions and returns the cached answer with a notice.
• Passes conversation context into the generator.
• Triggers auto-summarisation after every turn.
"""
from __future__ import annotations

from typing import Dict, Optional

from rag.generator import generate_answer, generate_standalone_question
from rag.memory import ConversationMemory
from rag.router import detect_tool, extract_date_info, extract_math_expression
from rag.tools import calculator_tool, date_tool, today_tool


class RAGPipeline:
    def __init__(self, retriever, memory: Optional[ConversationMemory] = None) -> None:
        self.retriever = retriever
        self.memory: ConversationMemory = memory or ConversationMemory()

    # ──────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────

    def ask(self, query: str, top_k: int = 3) -> Dict:
        """
        Process one turn.  Returns a dict compatible with the original
        app.py output schema, extended with memory fields.
        """
        # 1. Record the raw user question
        self.memory.add_user(query)

        # 2. Route detection (tools bypass RAG & memory)
        route = detect_tool(query)

        if route in ("calculator", "date", "today"):
            result = self._handle_tool(query, route)
            self.memory.add_assistant(result["answer"], route="tool")
            self.memory.maybe_summarise()
            return result

        # 3. Duplicate detection
        duplicate_answer = self.memory.find_duplicate(query)
        if duplicate_answer and self.memory.turn_count > 1:
            notice = (
                "🔁 **You asked a very similar question earlier.** "
                "Here is my previous answer — let me know if you'd "
                "like me to go deeper or approach it differently.\n\n"
                + duplicate_answer
            )
            self.memory.add_assistant(notice, route="rag")
            self.memory.maybe_summarise()
            return {
                "route": "rag",
                "answer": notice,
                "results": [],
                "duplicate": True,
                "conversation_turns": self.memory.turn_count,
                "was_summarised": False,
            }

        # 4. Query contextualisation — rewrite follow-ups for better retrieval
        conversation_context = self.memory.build_context_string()
        retrieval_query = query
        if self.memory.turn_count > 1 and conversation_context:
            retrieval_query = generate_standalone_question(query, conversation_context)

        # 5. Retrieve
        results = self.retriever.retrieve(query=retrieval_query, top_k=top_k)

        # 6. Generate with conversation context
        answer = generate_answer(
            question=query,
            contexts=results,
            conversation_context=conversation_context,
        )

        # 7. Record assistant turn
        sources = list({r["source"] for r in results})
        self.memory.add_assistant(answer, route="rag", sources=sources)

        # 8. Auto-summarise if needed
        was_summarised = self.memory.maybe_summarise()

        return {
            "route": "rag",
            "answer": answer,
            "results": results,
            "duplicate": False,
            "retrieval_query": retrieval_query,      # may differ from original query
            "conversation_turns": self.memory.turn_count,
            "was_summarised": was_summarised,
            "has_summary": bool(self.memory.summary),
        }

    # ──────────────────────────────────────────────────────────────
    # Tool handling (unchanged logic, extracted for clarity)
    # ──────────────────────────────────────────────────────────────

    def _handle_tool(self, query: str, route: str) -> Dict:
        if route == "calculator":
            expression = extract_math_expression(query)
            if not expression:
                return {
                    "route": "tool", "tool": "calculator", "input": None,
                    "answer": "I detected a calculator query, but could not extract a valid expression.",
                    "results": [],
                }
            result = calculator_tool(expression)
            answer = (
                f"The result is **{result['result']}**."
                if "result" in result
                else f"Calculator error: {result['error']}"
            )
            return {"route": "tool", "tool": "calculator", "input": expression,
                    "answer": answer, "results": []}

        if route == "date":
            date_info = extract_date_info(query)
            if not date_info:
                return {
                    "route": "tool", "tool": "date", "input": None,
                    "answer": "I detected a date query but could not parse a valid base date and offset.",
                    "results": [],
                }
            result = date_tool(base_date=date_info["base_date"], offset_days=date_info["offset_days"])
            answer = (
                f"The resulting date is **{result['result_date']}**."
                if "result_date" in result
                else f"Date tool error: {result['error']}"
            )
            return {"route": "tool", "tool": "date",
                    "input": {"base_date": date_info["base_date"], "offset_days": date_info["offset_days"]},
                    "answer": answer, "results": []}

        if route == "today":
            result = today_tool()
            return {"route": "tool", "tool": "today", "input": {},
                    "answer": f"Today's date is **{result['today']}**.", "results": []}

        return {"route": "rag", "answer": "Unknown route.", "results": []}