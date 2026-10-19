"""Command-line interface: run the self-improvement loop from a terminal."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from selfforge.core import improve

app = typer.Typer(add_completion=False, help="SelfForge — a self-improving LangGraph agent.")
console = Console()


@app.command()
def run(
    provider: str = typer.Option("offline", help="offline | groq"),
    model: str = typer.Option(None, help="Groq model id, e.g. llama-3.3-70b-versatile"),
    max_rounds: int = typer.Option(8),
    target: float = typer.Option(0.98),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON result"),
):
    """Run the agent and print its per-round improvement."""
    result = improve(provider=provider, model=model, max_rounds=max_rounds, target=target)

    if json_out:
        console.print_json(json.dumps(result.model_dump(mode="json")))
        raise typer.Exit()

    table = Table(title=f"SelfForge · {result.model}")
    table.add_column("Round", justify="right")
    table.add_column("Train", justify="right")
    table.add_column("Test", justify="right")
    table.add_column("Rules", justify="right")
    table.add_column("Learned")
    for r in result.rounds:
        table.add_row(
            str(r.round), f"{r.train_accuracy:.0%}", f"{r.test_accuracy:.0%}",
            str(r.n_rules), ", ".join(r.newly_learned) or "—",
        )
    console.print(table)
    console.print(
        f"[bold green]{result.start_test_accuracy:.0%} → {result.final_test_accuracy:.0%}[/] "
        f"held-out test  ([bold cyan]+{result.improvement*100:.0f} pts[/])"
    )


if __name__ == "__main__":
    app()
