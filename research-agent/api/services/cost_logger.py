"""
api/services/cost_logger.py

Estimates token usage and cost for each request.
Uses gpt-4o-mini pricing (placeholder — swap out for actual usage if needed).
"""
from __future__ import annotations
from api.models import TokenUsage

# gpt-4o-mini pricing (per 1M tokens, as of 2025)
_PRICE_IN_PER_1M  = 0.15   # input
_PRICE_OUT_PER_1M = 0.60   # output


def _rough_token_count(text: str) -> int:
    """~4 chars per token is a common approximation."""
    return max(1, len(text) // 4)


class CostLogger:
    """Stateless cost estimator. Each call is independent."""

    def estimate(
        self,
        question: str,
        answer: str,
        sub_questions: list[str],
    ) -> TokenUsage:
        """
        Rough estimate of tokens used across the full pipeline:
        decompose call + (N sub-question ReAct loops) + synthesize call.
        """
        # Decompose: ~50 token system + question → short list
        decompose_in  = 50 + _rough_token_count(question)
        decompose_out = sum(_rough_token_count(q) for q in sub_questions) + 20

        # ReAct per sub-q: system (~150) + question + search results (~300) → answer
        react_in  = len(sub_questions) * (150 + _rough_token_count(question) + 300)
        react_out = sum(_rough_token_count(q) for q in sub_questions) * 4

        # Synthesize: research block + system (~200) → final report
        research_block_tokens = react_out
        synth_in  = 200 + research_block_tokens
        synth_out = _rough_token_count(answer)

        prompt_tokens     = decompose_in + react_in + synth_in
        completion_tokens = decompose_out + react_out + synth_out
        total_tokens      = prompt_tokens + completion_tokens

        cost = (
            (prompt_tokens     / 1_000_000) * _PRICE_IN_PER_1M
            + (completion_tokens / 1_000_000) * _PRICE_OUT_PER_1M
        )

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost, 6),
        )