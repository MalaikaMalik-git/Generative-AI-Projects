"""
agent/react_loop.py — Core ReAct loop (Session 5: with error handling)
"""
from __future__ import annotations
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai.types.chat import ChatCompletionMessageParam
from agent.client       import client, OPENAI_MODEL
from agent.config       import MAX_STEPS
from agent.models       import Step, StepType, AgentResult
from agent.tracer       import print_step, print_header, print_footer
from agent.error_handler import safe_search, handle_max_steps
from tools.fetch         import fetch_page as _fetch_page

_TOOLS: dict[str, callable] = {
    "web_search": lambda args: safe_search(args["query"]),   # now uses safe_search
    "fetch_page": lambda args: _fetch_page(args["url"]),
}

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
            },
        },
    },
]

_SYSTEM = """\
You are a concise research assistant. Answer questions using web_search.

RULES — follow exactly:
1. Call web_search ONCE.
2. Read the snippets in the result.
3. Write your Final Answer based on those snippets.
4. Do NOT search again unless the first result says "Search unavailable".
5. Always end with: Final Answer: <2-4 sentence answer citing source URLs>
"""


def run(question: str, verbose: bool = True) -> AgentResult:
    """Run the ReAct loop for a given research question."""
    if verbose:
        print_header(question)

    steps: list[Step] = []
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": f"Research question: {question}"},
    ]

    for step_num in range(1, MAX_STEPS + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOL_SCHEMA,
                tool_choice="auto",
                max_tokens=600,
                temperature=0.2,
            )
        except Exception as e:
            # LLM call itself failed — return graceful error
            error_step = Step(StepType.ANSWER, f"LLM error: {e}")
            steps.append(error_step)
            if verbose:
                print_footer(success=False, step_count=step_num)
            return AgentResult(
                question=question,
                answer=f"Research failed due to LLM error: {e}",
                steps=steps,
                success=False,
                error="LLM_ERROR",
            )

        msg = response.choices[0].message

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                thought_text = msg.content.strip() if msg.content else f"Calling {fn_name}..."
                thought = Step(StepType.THOUGHT, thought_text)
                steps.append(thought)
                if verbose:
                    print_step(thought, step_num)

                action_text = f"{fn_name}({json.dumps(fn_args, ensure_ascii=False)})"
                action = Step(StepType.ACTION, action_text)
                steps.append(action)
                if verbose:
                    print_step(action, step_num)

                observation_text = (
                    _TOOLS[fn_name](fn_args)
                    if fn_name in _TOOLS
                    else f"Error: unknown tool '{fn_name}'"
                )
                observation = Step(StepType.OBSERVATION, observation_text)
                steps.append(observation)
                if verbose:
                    print_step(observation, step_num)

                messages.append(msg)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      observation_text,
                })

        else:
            text = (msg.content or "").strip()

            if "Final Answer:" in text:
                answer_text = text.split("Final Answer:", 1)[1].strip()
                answer_step = Step(StepType.ANSWER, answer_text)
                steps.append(answer_step)
                if verbose:
                    print_step(answer_step, step_num)
                    print_footer(success=True, step_count=step_num)

                return AgentResult(
                    question=question,
                    answer=answer_text,
                    steps=steps,
                    success=True,
                )

            thought = Step(StepType.THOUGHT, text)
            steps.append(thought)
            if verbose:
                print_step(thought, step_num)
            messages.append({"role": "assistant", "content": text})

    # Max steps exceeded — use error handler
    partial = handle_max_steps(question, steps)
    if verbose:
        print_footer(success=False, step_count=MAX_STEPS)

    return AgentResult(
        question=question,
        answer=partial,
        steps=steps,
        success=False,
        error="MAX_STEPS_EXCEEDED",
    )