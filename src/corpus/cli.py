from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner  # noqa: F401 — used via Live
from rich.text import Text

from corpus.ingestion import get_retriever, ingest_url
from corpus.reranker import rerank

app = typer.Typer(add_completion=False, help="Corpus knowledge base CLI.")
console = Console()


@app.command()
def add(source: str = typer.Argument(..., help="URL to ingest.")) -> None:
    """Ingest a source into the knowledge base."""
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


@app.callback(invoke_without_command=True)
def repl(ctx: typer.Context) -> None:
    """Open an interactive query REPL against the knowledge base."""
    if ctx.invoked_subcommand is not None:
        return

    with Live(Spinner("dots", text="Loading retriever…"), console=console, transient=True):
        retriever = get_retriever()

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

        with Live(Spinner("dots", text="Searching…"), console=console, transient=True):
            results = rerank(query, retriever.invoke(query))

        if not results:
            console.print("[dim]No results.[/dim]\n")
            continue

        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "")
            title = f"Result {i}" + (f"  [dim]{source}[/dim]" if source else "")
            console.print(Panel(Text(doc.page_content), title=title, border_style="blue"))

        console.print()
