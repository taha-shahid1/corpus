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

DB_PATH = str(Path("~/.corpus/corpus.db").expanduser())
