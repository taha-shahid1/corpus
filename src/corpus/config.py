from pathlib import Path

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DEVICE = "mps"

LANCEDB_URI = str(Path("~/.corpus/lancedb").expanduser())
LANCEDB_TABLE = "corpus_children"

DOCSTORE_PATH = Path("~/.corpus/parents").expanduser()

CHILD_CHUNK_SIZE = 200
