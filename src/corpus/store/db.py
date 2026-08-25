from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import sqlite_utils

from corpus.config import DB_PATH

_db_local = threading.local()


def get_db() -> sqlite_utils.Database:
    """Return a per-thread cached connection to the shared corpus SQLite file."""
    if not hasattr(_db_local, "conn"):
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _db_local.conn = sqlite_utils.Database(DB_PATH)
    return _db_local.conn


def ensure_table(db: sqlite_utils.Database, name: str, schema: dict, **create_kwargs) -> None:
    """Create `name` with `schema` if missing, tolerating races from concurrent workers."""
    if name in db.table_names():
        return
    try:
        db[name].create(schema, **create_kwargs)
    except sqlite3.OperationalError as exc:
        if "already exists" not in str(exc):
            raise
