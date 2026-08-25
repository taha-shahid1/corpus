from __future__ import annotations

from datetime import UTC, datetime

import sqlite_utils

from corpus.store.db import ensure_table, get_db

_TRACES = "traces"
_SPANS = "spans"


def _db() -> sqlite_utils.Database:
    db = get_db()
    ensure_table(
        db,
        _TRACES,
        {
            "trace_id": str,
            "started_at": str,
            "ended_at": str,
            "query": str,
            "status": str,
            "metadata": str,
        },
        pk="trace_id",
    )
    ensure_table(
        db,
        _SPANS,
        {
            "span_id": str,
            "trace_id": str,
            "parent_span_id": str,
            "name": str,
            "kind": str,
            "start_time": str,
            "end_time": str,
            "duration_ms": float,
            "input": str,
            "output": str,
            "metadata": str,
            "error": str,
        },
        pk="span_id",
        foreign_keys=[("trace_id", _TRACES)],
    )
    db[_SPANS].create_index(["trace_id"], if_not_exists=True)
    return db


def start_trace(trace_id: str, query: str, metadata: str = "{}") -> None:
    _db()[_TRACES].insert(
        {
            "trace_id": trace_id,
            "started_at": datetime.now(UTC).isoformat(),
            "ended_at": None,
            "query": query,
            "status": "running",
            "metadata": metadata,
        },
        replace=True,
    )


def end_trace(trace_id: str, status: str) -> None:
    _db()[_TRACES].update(
        trace_id,
        {"ended_at": datetime.now(UTC).isoformat(), "status": status},
    )


def insert_span(span: dict) -> None:
    _db()[_SPANS].insert(span, replace=True)


def list_traces(limit: int = 20) -> list[dict]:
    return list(_db()[_TRACES].rows_where(order_by="started_at desc", limit=limit))


def get_trace(trace_id: str) -> dict | None:
    try:
        return dict(_db()[_TRACES].get(trace_id))
    except sqlite_utils.db.NotFoundError:
        return None


def get_spans(trace_id: str) -> list[dict]:
    return list(
        _db()[_SPANS].rows_where("trace_id = ?", [trace_id], order_by="start_time asc")
    )
