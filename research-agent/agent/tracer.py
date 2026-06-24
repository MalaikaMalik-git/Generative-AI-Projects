"""
agent/tracer.py
Prints each ReAct step to the terminal with colour and structure.
Uses the `rich` library so the demo trace is readable at a glance.
"""
from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from agent.models import Step, StepType

console = Console()

# Colour + emoji for each step type
_STYLE: dict[StepType, tuple[str, str]] = {
    StepType.THOUGHT:     ("bold cyan",    "💭 Thought"),
    StepType.ACTION:      ("bold yellow",  "⚡ Action"),
    StepType.OBSERVATION: ("bold blue",    "🔍 Observation"),
    StepType.ANSWER:      ("bold green",   "✅ Answer"),
}


def print_step(step: Step, step_num: int | None = None) -> None:
    style, label = _STYLE[step.type]
    title = f"{label}" + (f"  [dim](step {step_num})[/dim]" if step_num else "")
    console.print(
        Panel(
            Text(step.content, style="white"),
            title=f"[{style}]{title}[/{style}]",
            border_style=style.split()[-1],   # last word is the colour
            padding=(0, 1),
        )
    )


def print_header(question: str) -> None:
    console.rule("[bold magenta]Research Agent — ReAct Loop[/bold magenta]")
    console.print(
        Panel(
            Text(question, style="bold white"),
            title="[magenta]Research Question[/magenta]",
            border_style="magenta",
        )
    )


def print_footer(success: bool, step_count: int) -> None:
    if success:
        console.rule(f"[bold green]Done — {step_count} steps[/bold green]")
    else:
        console.rule(f"[bold red]Stopped — {step_count} steps[/bold red]")