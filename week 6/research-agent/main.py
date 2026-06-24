"""
main.py — Full research pipeline with error handling
"""
from __future__ import annotations
import sys
import warnings

# Suppress urllib3 LibreSSL warning on macOS Python 3.9
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
try:
    import urllib3
    urllib3.disable_warnings()
except Exception:
    pass

from agent.decomposer     import decompose
from agent.react_loop     import run
from agent.error_handler  import safe_synthesize
from agent.tracer         import console
from rich.panel           import Panel
from rich.markdown        import Markdown


def main(question: str = None) -> None:
    if question is None:
        question = (
            input("Enter your research question: ").strip()
            or "What is the James Webb Space Telescope?"
        )

    # Step 1: Decompose
    console.rule("[bold magenta]Step 1 — Decomposing question[/bold magenta]")
    try:
        sub_questions = decompose(question)
    except Exception as e:
        console.print(f"[red]✗ Decomposition failed: {e}. Using original question.[/red]")
        sub_questions = [question]

    console.print(f"[dim]Broke into {len(sub_questions)} sub-questions:[/dim]")
    for i, q in enumerate(sub_questions, 1):
        console.print(f"  [cyan]{i}.[/cyan] {q}")

    # Step 2: ReAct loop per sub-question
    sub_results = []
    for i, sub_q in enumerate(sub_questions, 1):
        console.rule(f"[bold blue]Sub-question {i}/{len(sub_questions)}[/bold blue]")
        result = run(sub_q, verbose=True)
        sub_results.append({
            "question": sub_q,
            "answer":   result.answer,
            "success":  result.success,
        })

    # Step 3: Synthesize
    console.rule("[bold green]Step 3 — Synthesizing final report[/bold green]")
    console.print("[dim]Combining all findings into a structured report...[/dim]")
    report = safe_synthesize(question, sub_results)

    # Step 4: Print report
    console.rule("[bold green]Research Report[/bold green]")
    console.print(
        Panel(
            Markdown(report),
            title=f"[green]{question}[/green]",
            border_style="green",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or None
    main(question)