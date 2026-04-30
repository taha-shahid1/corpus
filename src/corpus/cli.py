from __future__ import annotations

import logging
import os
import sys
import warnings

# Suppress model-loading noise before any heavy imports land
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# LangGraph serialises AIMessages that have a `.parsed` field (from with_structured_output),
# which Pydantic can't round-trip cleanly. The warnings are harmless — silence them.
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module="pydantic",
)

try:
    from transformers.utils import logging as _hf_logging

    _hf_logging.set_verbosity_error()
    _hf_logging.disable_progress_bar()
except Exception:
    pass

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import typer
from langchain_core.messages import AIMessageChunk
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console, Group as RichGroup
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from corpus.agent.graph import build_graph
from corpus.ingestion import ingest_url
from corpus.retrieval.reranker import warmup as warmup_reranker
from corpus.storage import get_status, is_ingested

app = typer.Typer(add_completion=False, help="Corpus knowledge base CLI.")
console = Console()

_PROMPT_STYLE = PTStyle.from_dict({"prompt": "bold ansicyan"})

_NODE_LABELS: dict[str, str] = {
    "route": "Route",
    "plan": "Plan",
    "retrieve": "Retrieve",
    "grade": "Grade",
    "rewrite": "Rewrite",
    "generate": "Generate",
    "respond": "Respond",
}

# Nodes that stream the final answer token-by-token
_STREAMING_NODES: frozenset[str] = frozenset({"generate", "respond"})


def _node_detail(node_name: str, node_data: dict, retrieve_count: int) -> str:
    if node_name == "route":
        rt = node_data.get("route_type", "")
        return "knowledge lookup" if rt == "rag" else "conversational"
    if node_name == "plan":
        n = len(node_data.get("sub_questions", []))
        return f"{n} sub-question{'s' if n != 1 else ''}"
    if node_name == "retrieve":
        n = len(node_data.get("docs", []))
        return f"{n} doc{'s' if n != 1 else ''}"
    if node_name == "grade":
        n = len(node_data.get("docs", []))
        return f"{n}/{retrieve_count} relevant"
    if node_name == "rewrite":
        q = node_data.get("query", "")
        return f'"{q[:42]}…"' if len(q) > 42 else f'"{q}"'
    return ""


def _build_query_display(
    steps_done: list[tuple[str, str]],
    active_node: str | None,
    answer_chunks: list[str],
    generating: bool,
) -> RichGroup:
    parts: list = []

    for node_name, detail in steps_done:
        label = _NODE_LABELS.get(node_name, node_name.title())
        row = Text()
        row.append("  ✓  ", style="bold green")
        row.append(f"{label:<10}", style="green dim")
        if detail:
            row.append(f"  {detail}", style="dim")
        parts.append(row)

    if active_node:
        label = _NODE_LABELS.get(active_node, active_node.title())
        parts.append(Spinner("dots", text=f"  [cyan bold]{label}[/cyan bold]"))

    if answer_chunks:
        cursor = " ▋" if generating else ""
        parts.append(Text(""))
        parts.append(
            Panel(
                Markdown("".join(answer_chunks) + cursor),
                border_style="cyan",
                padding=(1, 2),
            )
        )

    return RichGroup(*parts)


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


def _render_sources(docs: list) -> None:
    """Print a deduplicated numbered sources list below the answer."""
    seen: dict[str, int] = {}
    for doc in docs:
        src = doc.metadata.get("source", "")
        if src and src not in seen:
            seen[src] = len(seen) + 1

    if not seen:
        return

    console.print("  [bold dim]Sources[/bold dim]")
    for src, n in seen.items():
        console.print(f"  [dim][{n}][/dim] [cyan]{src}[/cyan]")
    console.print()


@app.callback(invoke_without_command=True)
def repl(ctx: typer.Context) -> None:
    """Open an interactive agent REPL."""
    if ctx.invoked_subcommand is not None:
        return

    with Live(
        Spinner("dots", text="Loading agent…"),
        console=console,
        transient=True,
    ):
        graph = build_graph()
        warmup_reranker()

    console.print()
    console.print(Rule(" [bold cyan]Corpus[/bold cyan] ", style="cyan"))
    console.print("  [dim]Ask anything · [bold]↑↓[/bold] history · Ctrl+C to exit[/dim]\n")

    session: PromptSession = PromptSession(history=InMemoryHistory())

    while True:
        try:
            query = session.prompt("◆ ", style=_PROMPT_STYLE).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            sys.exit(0)

        if not query:
            continue

        console.print()

        steps_done: list[tuple[str, str]] = []
        active_node: str | None = "plan"
        answer_chunks: list[str] = []
        final_docs: list = []
        retrieve_count = 0

        try:
            with Live(
                _build_query_display(steps_done, active_node, answer_chunks, False),
                console=console,
                refresh_per_second=15,
                vertical_overflow="visible",
            ) as live:
                for mode, data in graph.stream(
                    {"query": query, "loop_count": 0},
                    stream_mode=["messages", "updates"],
                ):
                    if mode == "updates":
                        node_name = next(iter(data))
                        node_data = data[node_name]

                        if node_name == "retrieve":
                            retrieve_count = len(node_data.get("docs", []))
                        elif node_name == "grade":
                            final_docs = node_data.get("docs", [])
                        elif node_name in _STREAMING_NODES and not answer_chunks:
                            # fallback: non-streaming model emits full answer in updates
                            if ans := node_data.get("answer", ""):
                                answer_chunks.append(ans)

                        detail = _node_detail(node_name, node_data, retrieve_count)
                        steps_done.append((node_name, detail))

                        # infer the next active node from graph structure
                        if node_name == "route":
                            active_node = (
                                "plan"
                                if node_data.get("route_type") == "rag"
                                else "respond"
                            )
                        elif node_name == "plan":
                            active_node = "retrieve"
                        elif node_name == "retrieve":
                            active_node = "grade"
                        elif node_name == "grade":
                            # conditional edge — resolved by next event
                            active_node = None
                        elif node_name == "rewrite":
                            active_node = "plan"
                        elif node_name in _STREAMING_NODES:
                            active_node = None

                    elif mode == "messages":
                        chunk, meta = data
                        # Only accumulate AIMessageChunk tokens from answer-producing nodes.
                        # LangGraph also emits full HumanMessage/AIMessage objects when nodes
                        # write to the messages state key — filtering them out prevents the
                        # duplicated-answer bug ("…yet [1].heyHi. The provided…").
                        if (
                            meta.get("langgraph_node") in _STREAMING_NODES
                            and isinstance(chunk, AIMessageChunk)
                        ):
                            if active_node not in _STREAMING_NODES:
                                active_node = meta["langgraph_node"]
                            content = chunk.content
                            if isinstance(content, str):
                                answer_chunks.append(content)
                            elif isinstance(content, list):
                                for part in content:
                                    if isinstance(part, str):
                                        answer_chunks.append(part)
                                    elif isinstance(part, dict) and isinstance(
                                        part.get("text"), str
                                    ):
                                        answer_chunks.append(part["text"])

                    live.update(
                        _build_query_display(
                            steps_done,
                            active_node,
                            answer_chunks,
                            generating=(active_node in _STREAMING_NODES),
                        )
                    )

                # Final clean render — drop cursor and active spinner
                live.update(_build_query_display(steps_done, None, answer_chunks, False))

        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}\n")
            continue

        if not answer_chunks:
            console.print("[dim]No answer produced.[/dim]\n")
            continue

        _render_sources(final_docs)
