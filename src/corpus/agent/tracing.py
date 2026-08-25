from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from corpus.store.traces import end_trace, insert_span, start_trace

_MAX_TEXT_LEN = 4000


def new_trace_id() -> str:
    return uuid.uuid4().hex


def _dump(value: Any) -> str:
    """Serialize `value` to JSON, guaranteeing the result is always valid JSON.

    Truncating an already-serialized JSON string (as a plain string slice) can cut
    it mid-escape-sequence and produce invalid JSON. So truncation happens on the
    raw value *before* encoding — the truncated preview then gets escaped properly.
    """
    try:
        serialized = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        serialized = None

    if serialized is not None and len(serialized) <= _MAX_TEXT_LEN:
        return serialized

    raw = value if isinstance(value, str) else str(value)
    preview = raw[: _MAX_TEXT_LEN - 200]
    return json.dumps({"_truncated": True, "preview": preview}, ensure_ascii=False)


def _flatten_message(message) -> str:
    content = message.content
    text = content if isinstance(content, str) else str(content)
    return f"{message.type}: {text}"


class SQLiteTraceHandler(BaseCallbackHandler):
    """Captures every LangGraph node and LLM call as a span in the local trace store."""

    def __init__(self, trace_id: str, query: str) -> None:
        self.trace_id = trace_id
        self._lock = threading.Lock()
        self._pending: dict[UUID, dict] = {}
        start_trace(trace_id, query)

    def finish(self, status: str = "ok") -> None:
        end_trace(self.trace_id, status)

    # -- chain (LangGraph node) spans ----------------------------------

    def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs
    ):
        name = kwargs.get("name") or (serialized or {}).get("name") or "chain"
        self._start_span(run_id, parent_run_id, name, "chain", inputs)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._end_span(run_id, output=outputs)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._end_span(run_id, error=str(error))

    # -- LLM / chat-model spans -----------------------------------------

    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs
    ):
        name = kwargs.get("name") or (serialized or {}).get("name") or "llm"
        self._start_span(run_id, parent_run_id, name, "llm", prompts)

    def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        name = kwargs.get("name") or (serialized or {}).get("name") or "llm"
        flat = [
            [_flatten_message(m) for m in batch]
            for batch in messages
        ]
        self._start_span(run_id, parent_run_id, name, "llm", flat)

    def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id=None, tags=None, **kwargs):
        usage = self._extract_usage(response)
        chunks = [
            gen.text
            for batch in response.generations
            for gen in batch
            if getattr(gen, "text", None)
        ]
        text = "\n".join(chunks)
        self._end_span(run_id, output=text, extra_metadata={"usage": usage} if usage else None)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, tags=None, **kwargs):
        self._end_span(run_id, error=str(error))

    # -- internals --------------------------------------------------------

    def _start_span(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        name: str,
        kind: str,
        input_value: Any,
    ) -> None:
        with self._lock:
            self._pending[run_id] = {
                "span_id": run_id.hex,
                "trace_id": self.trace_id,
                "parent_span_id": parent_run_id.hex if parent_run_id else None,
                "name": name,
                "kind": kind,
                "start_time": datetime.now(UTC).isoformat(),
                "_t0": time.perf_counter(),
                "input": _dump(input_value),
            }

    def _end_span(
        self,
        run_id: UUID,
        output: Any = None,
        error: str | None = None,
        extra_metadata: dict | None = None,
    ) -> None:
        with self._lock:
            span = self._pending.pop(run_id, None)
        if span is None:
            return
        t0 = span.pop("_t0")
        span["end_time"] = datetime.now(UTC).isoformat()
        span["duration_ms"] = (time.perf_counter() - t0) * 1000
        span["output"] = _dump(output) if output is not None else None
        span["error"] = error
        span["metadata"] = _dump(extra_metadata or {})
        insert_span(span)

    @staticmethod
    def _extract_usage(response: LLMResult) -> dict | None:
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if usage:
            return dict(usage)
        for batch in response.generations:
            for gen in batch:
                meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if meta:
                    return dict(meta)
        return None
