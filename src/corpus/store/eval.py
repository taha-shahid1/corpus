from __future__ import annotations

from datetime import UTC, datetime

import sqlite_utils

from corpus.store.db import ensure_table, get_db

_RUNS = "eval_runs"
_CASES = "eval_cases"


def _db() -> sqlite_utils.Database:
    db = get_db()
    ensure_table(
        db,
        _RUNS,
        {
            "run_id": str,
            "started_at": str,
            "ended_at": str,
            "git_sha": str,
            "total": int,
            "passed": int,
            "pass_rate": float,
            "metadata": str,
        },
        pk="run_id",
    )
    ensure_table(
        db,
        _CASES,
        {
            "case_run_id": str,
            "run_id": str,
            "case_id": str,
            "category": str,
            "retrieval_hit": str,  # "true" | "false" | "n/a" — sqlite has no bool type
            "judge_correct": str,
            "passed": str,
            "trace_id": str,
            "reasoning": str,
        },
        pk="case_run_id",
        foreign_keys=[("run_id", _RUNS)],
    )
    db[_CASES].create_index(["run_id"], if_not_exists=True)
    return db


def start_run(run_id: str, git_sha: str | None, metadata: str = "{}") -> None:
    _db()[_RUNS].insert(
        {
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "ended_at": None,
            "git_sha": git_sha,
            "total": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "metadata": metadata,
        },
        replace=True,
    )


def finish_run(run_id: str, total: int, passed: int) -> None:
    _db()[_RUNS].update(
        run_id,
        {
            "ended_at": datetime.now(UTC).isoformat(),
            "total": total,
            "passed": passed,
            "pass_rate": (passed / total) if total else 0.0,
        },
    )


def insert_case(case: dict) -> None:
    _db()[_CASES].insert(case, replace=True)


def list_runs(limit: int = 20) -> list[dict]:
    return list(_db()[_RUNS].rows_where(order_by="started_at desc", limit=limit))


def get_run(run_id: str) -> dict | None:
    try:
        return dict(_db()[_RUNS].get(run_id))
    except sqlite_utils.db.NotFoundError:
        return None


def get_cases(run_id: str) -> list[dict]:
    return list(_db()[_CASES].rows_where("run_id = ?", [run_id], order_by="case_id asc"))
