"""Async queue + batched flush task for the upstream-request log store.

This is the analogue of ``a production proxy gateway/backend/ai_proxy/logging/service.py``:
the request hot path drops a :class:`LogEntry` onto an asyncio queue and
returns immediately; a single background task drains the queue in
batches and writes them to the SQLite store.

Why an asyncio queue and not just ``session.add`` from the handler?
Because we want zero latency cost on the proxied request: the FastAPI
handler shouldn't wait for SQLite's write lock before yielding the
response. The transport runs in the threadpool (sync httpx), so the
enqueue path actually bridges from a worker thread back to the loop via
``run_coroutine_threadsafe`` -- see :mod:`logging.transport`.

Public surface:

- :func:`enqueue_log` -- coroutine; drops an entry on the queue or warns
  on overflow.
- :func:`start_logging_service` / :func:`stop_logging_service` --
  lifespan hooks. Idempotent: calling stop on a never-started service is
  a no-op.
- :func:`drain_for_test` -- testing helper that synchronously flushes
  pending entries to the DB (no background task needed).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from decoding_sandbox.web.logging.db import RequestLog, get_session_factory
from decoding_sandbox.web.logging.masking import mask_headers, mask_sensitive_fields
from decoding_sandbox.web.logging.models import LogEntry

log = logging.getLogger("decoding_sandbox.web.logging.service")


# A bounded queue keeps a misbehaving upstream (or a misconfigured
# logging DB) from chewing through memory. 10k entries at ~10 kB each
# is ~100 MB, which is plenty of head-room before we start dropping --
# the writer batches 50/flush so even a sustained 100 RPS would have to
# fall ~200x behind the writer to fill this queue.
_QUEUE_MAXSIZE = 10000

# Module-level singletons; only meaningful when the lifespan hook is
# active.
_queue: asyncio.Queue[LogEntry] | None = None
_flush_task: asyncio.Task[None] | None = None


def get_queue() -> asyncio.Queue[LogEntry] | None:
    """Return the live queue (or ``None`` if the service isn't running)."""
    return _queue


async def enqueue_log(entry: LogEntry) -> None:
    """Drop ``entry`` on the queue; no-op if the service isn't running.

    On overflow we log a warning rather than block the producer -- the
    transport runs on the same event loop the handler does, and a
    blocked enqueue would hold up the user's response. a production proxy gateway takes
    the same trade-off; the symptom is "I lost some log rows", not
    "my API stalled".
    """
    if _queue is None:
        return
    try:
        _queue.put_nowait(entry)
    except asyncio.QueueFull:
        log.warning("dsbx-web log queue full (>%d entries); dropping one", _QUEUE_MAXSIZE)


async def _flush_loop(batch_size: int, flush_interval: float) -> None:
    """Drain the queue in batches and write rows to SQLite.

    Wakes on either (a) the first entry showing up after an idle period
    (via ``await queue.get()``), or (b) the flush interval elapsing
    (``asyncio.wait_for``). Inside the wake, drain up to ``batch_size``
    pending entries opportunistically -- the writer cost is dominated
    by the transaction, not the row count, so batching cheaply trades
    a little staleness for a lot less commit overhead.

    On shutdown (``CancelledError``) we flush whatever's left and then
    propagate the cancellation so the lifespan exit path completes.
    """
    assert _queue is not None
    session_factory = get_session_factory()
    if session_factory is None:
        log.error("dsbx-web: log flush loop started without a DB engine; bailing")
        return

    while True:
        entries: list[LogEntry] = []
        try:
            try:
                first = await asyncio.wait_for(_queue.get(), timeout=flush_interval)
                entries.append(first)
            except asyncio.TimeoutError:
                continue
            while len(entries) < batch_size:
                try:
                    entries.append(_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            await _write_batch(session_factory, entries)
        except asyncio.CancelledError:
            while not _queue.empty():
                try:
                    entries.append(_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if entries:
                try:
                    await _write_batch(session_factory, entries)
                except Exception:  # noqa: BLE001
                    log.exception("dsbx-web: shutdown log flush failed")
            raise
        except Exception:  # noqa: BLE001
            # Don't let one bad batch kill the writer task; SQLite is
            # local so the typical failure is "schema drift after a
            # rename" which fixes itself when the operator deletes
            # ``logs.db``. Log and keep going.
            log.exception("dsbx-web: log flush batch errored (%d entries)", len(entries))


async def _write_batch(
    session_factory: async_sessionmaker[AsyncSession], entries: list[LogEntry]
) -> None:
    """Persist a batch of :class:`LogEntry` Pydantic models to SQLite.

    We translate from the in-memory model to the ORM row here (and apply
    secret masking on header/body fields) rather than in :func:`enqueue_log`
    so the producer path stays trivially fast and any masking cost
    amortizes over the batch.
    """
    if not entries:
        return
    async with session_factory() as session:
        for entry in entries:
            session.add(_entry_to_row(entry))
        await session.commit()
    log.debug("dsbx-web: wrote %d log entries", len(entries))


def _entry_to_row(entry: LogEntry) -> RequestLog:
    """Convert a :class:`LogEntry` Pydantic model to a SQLAlchemy ``RequestLog``."""
    return RequestLog(
        id=str(entry.id),
        timestamp=entry.timestamp,
        backend_name=entry.backend_name,
        backend_family=entry.backend_family,
        provider_name=entry.provider_name,
        method=entry.method,
        upstream_url=entry.upstream_url,
        upstream_path=entry.upstream_path,
        request_headers=mask_headers(entry.request_headers),
        request_body=mask_sensitive_fields(entry.request_body),
        request_body_text=entry.request_body_raw,
        response_status_code=entry.response_status_code,
        response_headers=mask_headers(entry.response_headers),
        response_body=entry.response_body,
        response_body_text=entry.response_body_raw,
        stream_chunks=entry.stream_chunks,
        is_streaming=bool(entry.is_streaming),
        latency_ms=entry.latency_ms,
        ttft_ms=entry.ttft_ms,
        prompt_tokens=entry.prompt_tokens,
        completion_tokens=entry.completion_tokens,
        total_tokens=entry.total_tokens,
        model_resolved=entry.model_resolved,
        completion_text=entry.completion_text,
        stop_reason=entry.stop_reason,
        error_message=entry.error_message,
    )


def start_logging_service(
    *, batch_size: int = 50, flush_interval: float = 5.0
) -> asyncio.Task[None]:
    """Bring the queue up and start the background flush task.

    Must be called from an active event loop (the lifespan startup
    hook). Idempotent -- a second call is a no-op that returns the
    already-running task so callers don't have to special-case
    re-entry during a hot reload.
    """
    global _queue, _flush_task
    if _flush_task is not None and not _flush_task.done():
        return _flush_task
    _queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    _flush_task = asyncio.create_task(
        _flush_loop(batch_size=batch_size, flush_interval=flush_interval),
        name="dsbx-web-log-flush",
    )
    log.info("dsbx-web: log flush task started (batch=%d, interval=%.1fs)",
             batch_size, flush_interval)
    return _flush_task


async def stop_logging_service() -> None:
    """Cancel the flush task and await its final shutdown flush."""
    global _flush_task, _queue
    if _flush_task is None:
        return
    _flush_task.cancel()
    with suppress(asyncio.CancelledError):
        await _flush_task
    _flush_task = None
    _queue = None
    log.info("dsbx-web: log flush task stopped")


async def drain_for_test() -> int:
    """Synchronously flush the queue. Intended for tests only.

    Returns the number of entries written. Doesn't touch the background
    task, so it's safe to call from a test that ALSO has the service
    running -- it just races the flush loop, and SQLite's per-row
    write-ordering makes both paths converge to the same end state.
    """
    if _queue is None:
        return 0
    session_factory = get_session_factory()
    if session_factory is None:
        return 0
    entries: list[LogEntry] = []
    while not _queue.empty():
        try:
            entries.append(_queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    if entries:
        await _write_batch(session_factory, entries)
    return len(entries)


__all__ = [
    "drain_for_test",
    "enqueue_log",
    "get_queue",
    "start_logging_service",
    "stop_logging_service",
]
