from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from watchdog.events import (
    FileCreatedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from corpus.config import WATCHER_DEBOUNCE_SECONDS, WATCHER_MAX_WORKERS
from corpus.ingestion.pipeline import (
    ingest_md,
    ingest_pdf,
    rebuild_fts_index,
    remove_source_embeddings,
)
from corpus.storage import get_file_hash

logger = logging.getLogger(__name__)

_SUPPORTED: frozenset[str] = frozenset({".md", ".pdf"})
_INGEST_FN = {
    ".md": ingest_md,
    ".pdf": ingest_pdf,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _ingest_file(
    path: Path,
    *,
    rebuild_index: bool = True,
    on_status: Callable[[str], None] | None = None,
) -> bool:
    """Ingest or re-ingest a single file, skipping if the content is unchanged.

    Returns True if the file was actually ingested, False if it was skipped.
    """
    ext = path.suffix.lower()
    if ext not in _INGEST_FN:
        return False

    source = str(path)
    if on_status:
        on_status(f"checking {path.name}")
    try:
        current_hash = _sha256(path)
    except OSError:
        if on_status:
            on_status(f"skipped {path.name} (file disappeared)")
        return False  # File disappeared between event and processing

    stored_hash = get_file_hash(source)
    if stored_hash is not None:
        if stored_hash == current_hash:
            logger.debug("Unchanged, skipping: %s", path.name)
            if on_status:
                on_status(f"skipped {path.name} (unchanged)")
            return False
        logger.info("Content changed, re-indexing: %s", path.name)
        if on_status:
            on_status(f"changed {path.name}, re-indexing")
        remove_source_embeddings(source)

    logger.info("Ingesting: %s", path.name)
    if on_status:
        on_status(f"ingesting {path.name}")
    _INGEST_FN[ext](source, force=True, file_hash=current_hash, rebuild_index=rebuild_index)
    if on_status:
        on_status(f"ingested {path.name}")
    return True


class _EventHandler(FileSystemEventHandler):
    def __init__(
        self,
        executor: ThreadPoolExecutor,
        debounce: float,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._executor = executor
        self._debounce = debounce
        self._on_status = on_status
        self._pending: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _schedule(self, raw_path: str) -> None:
        path = Path(raw_path)
        if path.suffix.lower() not in _SUPPORTED or not path.is_file():
            return
        key = str(path.resolve())
        with self._lock:
            existing = self._pending.pop(key, None)
            if existing:
                existing.cancel()
            timer = threading.Timer(self._debounce, self._dispatch, args=(key,))
            self._pending[key] = timer
            timer.start()
        if self._on_status:
            self._on_status(f"queued {path.name}")

    def _dispatch(self, key: str) -> None:
        with self._lock:
            self._pending.pop(key, None)
        # Live watch events are infrequent — rebuild the index each time.
        self._executor.submit(
            _ingest_file, Path(key), rebuild_index=True, on_status=self._on_status
        )

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            self._schedule(event.dest_path)


class FolderWatcher:
    """Watch one or more folders and automatically ingest new or updated documents."""

    def __init__(
        self,
        folders: list[Path],
        *,
        workers: int = WATCHER_MAX_WORKERS,
        debounce: float = WATCHER_DEBOUNCE_SECONDS,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._folders = [f.resolve() for f in folders]
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="corpus-ingest")
        self._on_status = on_status
        self._handler = _EventHandler(self._executor, debounce, on_status=on_status)
        self._observer = Observer()

    def start(self) -> None:
        if self._on_status:
            self._on_status("running initial scan")
        self._initial_scan()
        for folder in self._folders:
            self._observer.schedule(self._handler, str(folder), recursive=True)
        self._observer.start()
        if self._on_status:
            self._on_status("watcher ready")
        logger.info(
            "Watching %d folder(s) with %d worker(s)",
            len(self._folders),
            self._executor._max_workers,
        )

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()
        self._executor.shutdown(wait=True)

    def _initial_scan(self) -> None:
        candidates = [
            p
            for folder in self._folders
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in _SUPPORTED
        ]
        if not candidates:
            if self._on_status:
                self._on_status("initial scan complete (no supported files)")
            return

        logger.info("Initial scan: %d file(s) found", len(candidates))
        if self._on_status:
            self._on_status(f"initial scan found {len(candidates)} file(s)")
        # Suppress per-file FTS rebuilds; do one rebuild after the whole batch.
        futures = {
            self._executor.submit(
                _ingest_file, p, rebuild_index=False, on_status=self._on_status
            ): p
            for p in candidates
        }
        any_ingested = False
        completed = 0
        for future in as_completed(futures):
            p = futures[future]
            try:
                if future.result():
                    any_ingested = True
                completed += 1
                if self._on_status:
                    self._on_status(f"initial scan progress {completed}/{len(candidates)}")
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", p.name, exc)
                if self._on_status:
                    self._on_status(f"failed {p.name}: {exc}")

        if any_ingested:
            if self._on_status:
                self._on_status("rebuilding search index")
            rebuild_fts_index()
            if self._on_status:
                self._on_status("initial scan complete")
        elif self._on_status:
            self._on_status("initial scan complete (no new changes)")
