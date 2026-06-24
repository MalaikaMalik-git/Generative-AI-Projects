"""
hello_world.py
Session 1 verification script.

Run:
    python hello_world.py

What it checks:
  1. .env loads correctly (OPENAI_API_KEY is present)
  2. OpenAI client connects and returns a response
  3. The model we plan to use is accessible
"""
import sys
import os

# Make sure we can import from the project root
sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def run():
    console.rule("[bold blue]Research Agent — Session 1 Hello World[/bold blue]")

    # ── Step 1: config ────────────────────────────────────────────────────────
    try:
        from agent.config import OPENAI_MODEL, MAX_STEPS
        console.print(f"[green]✓[/green] Config loaded  |  model=[bold]{OPENAI_MODEL}[/bold]  max_steps={MAX_STEPS}")
    except EnvironmentError as e:
        console.print(f"[red]✗ Config error:[/red] {e}")
        sys.exit(1)

    # ── Step 2: client ────────────────────────────────────────────────────────
    try:
        from agent.client import client
        console.print("[green]✓[/green] OpenAI client initialised")
    except Exception as e:
        console.print(f"[red]✗ Client error:[/red] {e}")
        sys.exit(1)

    # ── Step 3: live API call ─────────────────────────────────────────────────
    console.print("\n[dim]Sending hello-world message to OpenAI...[/dim]")

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research agent assistant. "
                        "Reply in exactly one sentence."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Say hello and confirm you are ready to help "
                        "with research tasks."
                    ),
                },
            ],
            max_tokens=60,
        )
    except Exception as e:
        console.print(f"[red]✗ API call failed:[/red] {e}")
        sys.exit(1)

    reply = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens

    console.print(
        Panel(
            Text(reply, style="bold white"),
            title=f"[green]✓ Response from {OPENAI_MODEL}[/green]",
            subtitle=f"[dim]{tokens_used} tokens used[/dim]",
            border_style="green",
        )
    )

    console.rule("[bold green]Session 1 complete — all checks passed[/bold green]")


if __name__ == "__main__":
    run()