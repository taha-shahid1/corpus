# Corpus

A CLI for turning PDFs, Markdown, and web pages into something you can actually query. Everything stays on disk under your home directory—vectors in LanceDB, parent chunks in a local docstore, and a SQLite ledger of what you’ve ingested.

<div align="center">

![Corpus: adding a document and asking the REPL a question](demo.mp4)

</div>

---

## How it works

Ingestion walks each source through parsers (URLs via trafilatura, PDFs via PyMuPDF, Markdown via Unstructured), splits text into parent/child chunks, and embeds the children with [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). Retrieval uses a parent-document setup: search hits the small chunks, but the model reads the larger parent spans they came from. LanceDB runs hybrid search by default (dense vectors plus BM25), and a cross-encoder reranker filters noise before answers are drafted.

The chat side is a LangGraph agent: it decides whether you’re asking for grounded facts from the corpus or just chatting, plans sub-questions for trickier prompts, retrieves and grades passages, and can rewrite the query once or twice if nothing looks relevant. Answers stream in the terminal with a thin trace of which graph nodes ran.

## Requirements

- **Python 3.13+**
- **PyTorch** (pulled in via dependencies; on Apple Silicon it will use MPS when available)
- **An LLM backend** — either API keys for a hosted provider, or [Ollama](https://ollama.com/) running locally

The first run downloads embedding and reranker weights from Hugging Face (hundreds of MB). The CLI defaults to offline Hub access after that; if you need a fresh download, run with `HF_HUB_OFFLINE=0`.

## Install

### From the repo (development)

From the repo root:

```bash
uv sync
# or: pip install -e .
```

The package exposes a `corpus` command.

### As a CLI with uv (from a wheel)

Build a wheel into `dist/`, then install it as an isolated tool so `corpus` is on your PATH:

```bash
uv build
uv tool install ./dist/corpus-0.1.0-py3-none-any.whl
```

After a version bump in `pyproject.toml`, the wheel filename changes—adjust the path or use a glob such as `./dist/corpus-*-py3-none-any.whl`.

To upgrade, rebuild the wheel and run `uv tool install` again with the new file (uv replaces the existing tool install).

## Usage

**Ingest something**

```bash
corpus add https://example.com/article
corpus add ./paper.pdf
corpus add ./notes.md
```

Already-ingested sources are skipped unless you change the file (the watcher path hashes local files and re-indexes on edits).

**Ask questions**

Running `corpus` with no subcommand opens the interactive REPL: arrow keys scroll command history, Ctrl+C exits. For factual prompts it pulls from your corpus; for small talk it answers without pretending it saw a document.

**Inspect what’s indexed**

```bash
corpus status
```

**Watch folders**

```bash
corpus watch ./research ./inbox
```

New or modified `.md` / `.pdf` files are picked up after a short debounce. Use `-w` / `--workers` if you want more parallel ingest threads.

## LLM providers

Set `CORPUS_LLM_PROVIDER` to one of `anthropic`, `openai`, `gemini`, or `ollama`. If you leave it unset, the app picks the first provider it can authenticate: Anthropic (`ANTHROPIC_API_KEY`), Google (`GOOGLE_API_KEY`), OpenAI (`OPENAI_API_KEY`), then falls back to Ollama at `http://localhost:11434`.

Model names are overridden with env vars if your account uses different ids—see `src/corpus/config.py` for the full list (`CORPUS_OPENAI_FAST_MODEL`, `CORPUS_ANTHROPIC_STRONG_MODEL`, etc.).

## Data layout

| Path | Purpose |
|------|---------|
| `~/.corpus/corpus.db` | SQLite: ingested sources, timestamps, file hashes |
| `~/.corpus/lancedb/` | Vector + FTS index |
| `~/.corpus/parents/` | Docstore for parent chunks |

Removing files manually isn’t recommended; use the ingestion APIs or wipe `~/.corpus` if you want a clean slate.

## Configuration snippets

- **`CORPUS_EMBEDDING_DEVICE` / `CORPUS_RERANKER_DEVICE`** — force `cpu`, `mps`, or `cuda` instead of auto-detection.
- **`CORPUS_RERANKER_MIN_SCORE`** — floor for reranker scores before the agent tries a query rewrite (default `0.0`).

## Development

```bash
uv run ruff check .
```

---

MIT License — see `LICENSE`.
