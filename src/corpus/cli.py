from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner  # noqa: F401 — used via Live
from rich.table import Table
from rich.text import Text

from corpus.agent.graph import build_graph
from corpus.ingestion import ingest_url
from corpus.storage import get_status, is_ingested

app = typer.Typer(add_completion=False, help="Corpus knowledge base CLI.")
console = Console()


@app.command()
def add(source: str = typer.Argument(..., help="URL to ingest.")) -> None:
    """Ingest a source into the knowledge base."""
    if is_ingested(source):
        console.print(f"[yellow]Already ingested:[/yellow] [cyan]{source}[/cyan]")
        return

    with Live(
        Spinner("dots", text=f"Ingesting [cyan]{source}[/cyan]…"),
        console=console,
        transient=True,
    ):
        try:
            ingest_url(source)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)

    console.print(f"[green]✓[/green] Ingested [cyan]{source}[/cyan]")


@app.command()
def status() -> None:
    """Show all ingested sources."""
    rows = get_status()
    if not rows:
        console.print("[dim]Nothing ingested yet.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Source")
    table.add_column("Docs", justify="right")
    table.add_column("Ingested")

    for row in rows:
        table.add_row(row["source"], str(row["doc_count"]), row["ingested_at"])

    console.print(table)


@app.callback(invoke_without_command=True)
def repl(ctx: typer.Context) -> None:
    """Open an interactive agent REPL."""
    if ctx.invoked_subcommand is not None:
        return

    with Live(Spinner("dots", text="Loading agent…"), console=console, transient=True):
        graph = build_graph()

    console.print(Rule("[bold]Corpus[/bold]"))
    console.print(
        "Type a query and press [bold]Enter[/bold]. [dim]Ctrl+C or Ctrl+D to exit.[/dim]\n"
    )

    while True:
        try:
            query = input("[query] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            sys.exit(0)

        if not query:
            continue

        with Live(Spinner("dots", text="Thinking…"), console=console, transient=True):
            result = graph.invoke({"query": query, "loop_count": 0})

        answer = result.get("answer", "")
        if not answer:
            console.print("[dim]No answer produced.[/dim]\n")
            continue

        console.print(Panel(Text(answer), title="Answer", border_style="green"))
        console.print()
