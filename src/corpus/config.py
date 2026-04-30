import os
from pathlib import Path

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DEVICE = "mps"

LANCEDB_URI = str(Path("~/.corpus/lancedb").expanduser())
LANCEDB_TABLE = "corpus_children"

DOCSTORE_PATH = Path("~/.corpus/parents").expanduser()

CHILD_CHUNK_SIZE = 200
EMBEDDING_BATCH_SIZE = 32

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "mps"
RERANKER_TOP_K = 5

AGENT_MAX_LOOPS = 2

OPENAI_FAST_MODEL = os.getenv("CORPUS_OPENAI_FAST_MODEL", "gpt-5.2")
OPENAI_STRONG_MODEL = os.getenv("CORPUS_OPENAI_STRONG_MODEL", "gpt-5.2")

ANTHROPIC_FAST_MODEL = os.getenv("CORPUS_ANTHROPIC_FAST_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_STRONG_MODEL = os.getenv("CORPUS_ANTHROPIC_STRONG_MODEL", "claude-sonnet-4-5")

GEMINI_FAST_MODEL = os.getenv("CORPUS_GEMINI_FAST_MODEL", "gemini-2.0-flash")
GEMINI_STRONG_MODEL = os.getenv("CORPUS_GEMINI_STRONG_MODEL", "gemini-2.5-pro-preview-05-06")

OLLAMA_FAST_MODEL = os.getenv("CORPUS_OLLAMA_FAST_MODEL", "llama3.2")
OLLAMA_STRONG_MODEL = os.getenv("CORPUS_OLLAMA_STRONG_MODEL", "llama3.1:70b")
OLLAMA_BASE_URL = os.getenv("CORPUS_OLLAMA_BASE_URL", "http://localhost:11434")

DB_PATH = str(Path("~/.corpus/corpus.db").expanduser())
