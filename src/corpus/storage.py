from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import sqlite_utils

from corpus.config import DB_PATH

_TABLE = "ingested_sources"


def _db() -> sqlite_utils.Database:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite_utils.Database(DB_PATH)
    if _TABLE not in db.table_names():
        db[_TABLE].create(
            {"source": str, "doc_count": int, "ingested_at": str},
            pk="source",
        )
    return db


def is_ingested(source: str) -> bool:
    try:
        _db()[_TABLE].get(source)
        return True
    except sqlite_utils.db.NotFoundError:
        return False


def mark_ingested(source: str, doc_count: int) -> None:
    _db()[_TABLE].insert(
        {
            "source": source,
            "doc_count": doc_count,
            "ingested_at": datetime.now(UTC).isoformat(),
        },
        replace=True,
    )


def get_status() -> list[dict]:
    return list(_db()[_TABLE].rows_where(order_by="ingested_at desc"))
