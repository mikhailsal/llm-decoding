"""FastAPI router exposing the upstream-request log store to the browser.

Endpoints (all behind ``require_bearer``):

- ``GET  /api/v1/logs``            -- paginated list of trimmed rows
- ``GET  /api/v1/logs/stats``      -- counts + aggregates
- ``GET  /api/v1/logs/search``     -- LIKE across a handful of columns
- ``GET  /api/v1/logs/{log_id}``   -- one full row including bodies + chunks
- ``DELETE /api/v1/logs``          -- delete rows by id or by timestamp

The router is wired in :func:`dsbx.web.app.make_web_app` only
when ``[web.logging].enabled`` is true; with logging disabled the import
is skipped entirely so the CLI install keeps working without SQLAlchemy.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import delete, desc, func, or_, select

from dsbx.web.logging.db import RequestLog, get_session_factory

log = logging.getLogger("dsbx.web.logs_api")


# --------------------------------------------------------------------------- #
# Pydantic wire shapes
# --------------------------------------------------------------------------- #
class LogSummary(BaseModel):
    """Trimmed row for the list view -- no big bodies on this shape."""

    id: str
    timestamp: dt.datetime
    backend_name: str
    backend_family: str
    provider_name: str | None
    method: str
    upstream_path: str
    response_status_code: int | None
    is_streaming: bool
    latency_ms: float | None
    ttft_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    model_resolved: str | None
    completion_text: str | None
    stop_reason: str | None
    error_message: str | None


class LogDetail(LogSummary):
    """Full row -- includes the big columns the detail panel renders."""

    upstream_url: str
    request_headers: dict[str, Any] | None
    request_body: Any
    request_body_text: str | None
    response_headers: dict[str, Any] | None
    response_body: Any
    response_body_text: str | None
    stream_chunks: list[Any] | None


class LogListResponse(BaseModel):
    """Page envelope: rows + a cursor for the next page."""

    items: list[LogSummary]
    next_cursor: str | None
    has_more: bool


class LogStats(BaseModel):
    total: int
    streaming: int
    non_streaming: int
    error_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_latency_ms: float | None
    avg_ttft_ms: float | None


class DeleteResult(BaseModel):
    deleted: int


# --------------------------------------------------------------------------- #
# ORM -> Pydantic
# --------------------------------------------------------------------------- #
def _row_to_summary(row: RequestLog) -> LogSummary:
    return LogSummary(
        id=row.id,
        timestamp=row.timestamp,
        backend_name=row.backend_name,
        backend_family=row.backend_family,
        provider_name=row.provider_name,
        method=row.method,
        upstream_path=row.upstream_path,
        response_status_code=row.response_status_code,
        is_streaming=row.is_streaming,
        latency_ms=row.latency_ms,
        ttft_ms=row.ttft_ms,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        total_tokens=row.total_tokens,
        model_resolved=row.model_resolved,
        completion_text=_truncate(row.completion_text, 240),
        stop_reason=row.stop_reason,
        error_message=row.error_message,
    )


def _row_to_detail(row: RequestLog) -> LogDetail:
    return LogDetail(
        id=row.id,
        timestamp=row.timestamp,
        backend_name=row.backend_name,
        backend_family=row.backend_family,
        provider_name=row.provider_name,
        method=row.method,
        upstream_path=row.upstream_path,
        upstream_url=row.upstream_url,
        response_status_code=row.response_status_code,
        is_streaming=row.is_streaming,
        latency_ms=row.latency_ms,
        ttft_ms=row.ttft_ms,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        total_tokens=row.total_tokens,
        model_resolved=row.model_resolved,
        completion_text=row.completion_text,
        stop_reason=row.stop_reason,
        error_message=row.error_message,
        request_headers=row.request_headers,
        request_body=row.request_body,
        request_body_text=row.request_body_text,
        response_headers=row.response_headers,
        response_body=row.response_body,
        response_body_text=row.response_body_text,
        stream_chunks=row.stream_chunks,
    )


def _truncate(s: str | None, max_len: int) -> str | None:
    if s is None:
        return None
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "\u2026"


# --------------------------------------------------------------------------- #
# Router factory
# --------------------------------------------------------------------------- #
def make_logs_router(require_bearer) -> APIRouter:
    """Build the ``/api/v1/logs`` router gated behind ``require_bearer``.

    Returning a fresh router from a factory matches the rest of the web
    layer (auth dependency is built in :func:`make_web_app`, not at
    import time); it also keeps the imports lazy enough that
    ``logs_api`` isn't pulled in unless logging is enabled.
    """
    router = APIRouter(
        prefix="/api/v1/logs",
        tags=["logs"],
        dependencies=[Depends(require_bearer)],
    )

    def _require_session_factory():
        sf = get_session_factory()
        if sf is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="upstream-request log store is not initialized",
            )
        return sf

    @router.get("", response_model=LogListResponse)
    async def list_logs(
        cursor: str | None = Query(
            default=None,
            description=(
                "ISO-8601 timestamp returned by the previous page's "
                "``next_cursor``. Rows STRICTLY older than this are "
                "returned. Omit to start from the most recent."
            ),
        ),
        limit: int = Query(default=50, ge=1, le=200),
        backend: str | None = Query(default=None),
        provider: str | None = Query(default=None),
        status_code: int | None = Query(default=None, alias="status_code"),
        is_error: bool | None = Query(
            default=None,
            description="Filter to error rows only (status >= 400 OR error_message set).",
        ),
        since: dt.datetime | None = Query(default=None),
    ) -> LogListResponse:
        """Return one page of log rows ordered newest-first."""
        sf = _require_session_factory()
        async with sf() as session:
            stmt = select(RequestLog).order_by(desc(RequestLog.timestamp))
            if backend:
                stmt = stmt.where(RequestLog.backend_name == backend)
            if provider:
                stmt = stmt.where(RequestLog.provider_name == provider)
            if status_code is not None:
                stmt = stmt.where(RequestLog.response_status_code == status_code)
            if is_error is True:
                stmt = stmt.where(
                    or_(
                        RequestLog.response_status_code >= 400,
                        RequestLog.error_message.isnot(None),
                    )
                )
            if since is not None:
                stmt = stmt.where(RequestLog.timestamp >= since)
            if cursor is not None:
                try:
                    cursor_ts = dt.datetime.fromisoformat(cursor)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"bad cursor: {exc}",
                    ) from exc
                stmt = stmt.where(RequestLog.timestamp < cursor_ts)
            stmt = stmt.limit(limit + 1)
            result = await session.execute(stmt)
            rows = list(result.scalars().all())

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]
        next_cursor = rows[-1].timestamp.isoformat() if has_more and rows else None
        return LogListResponse(
            items=[_row_to_summary(r) for r in rows],
            next_cursor=next_cursor,
            has_more=has_more,
        )

    @router.get("/stats", response_model=LogStats)
    async def stats() -> LogStats:
        """Aggregate counters useful for the dashboard header."""
        sf = _require_session_factory()
        async with sf() as session:
            total = await session.scalar(select(func.count(RequestLog.id)))
            streaming = await session.scalar(
                select(func.count(RequestLog.id)).where(RequestLog.is_streaming.is_(True))
            )
            error_count = await session.scalar(
                select(func.count(RequestLog.id)).where(
                    or_(
                        RequestLog.response_status_code >= 400,
                        RequestLog.error_message.isnot(None),
                    )
                )
            )
            prompt_sum = await session.scalar(
                select(func.coalesce(func.sum(RequestLog.prompt_tokens), 0))
            )
            completion_sum = await session.scalar(
                select(func.coalesce(func.sum(RequestLog.completion_tokens), 0))
            )
            avg_lat = await session.scalar(select(func.avg(RequestLog.latency_ms)))
            avg_ttft = await session.scalar(select(func.avg(RequestLog.ttft_ms)))
        total_int = int(total or 0)
        return LogStats(
            total=total_int,
            streaming=int(streaming or 0),
            non_streaming=total_int - int(streaming or 0),
            error_count=int(error_count or 0),
            total_prompt_tokens=int(prompt_sum or 0),
            total_completion_tokens=int(completion_sum or 0),
            avg_latency_ms=float(avg_lat) if avg_lat is not None else None,
            avg_ttft_ms=float(avg_ttft) if avg_ttft is not None else None,
        )

    @router.get("/search", response_model=LogListResponse)
    async def search(
        q: str = Query(..., min_length=1, max_length=200),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> LogListResponse:
        """LIKE search across upstream_url, model, error, and body text."""
        sf = _require_session_factory()
        like = f"%{q}%"
        async with sf() as session:
            stmt = (
                select(RequestLog)
                .where(
                    or_(
                        RequestLog.upstream_url.like(like),
                        RequestLog.upstream_path.like(like),
                        RequestLog.model_resolved.like(like),
                        RequestLog.error_message.like(like),
                        RequestLog.request_body_text.like(like),
                        RequestLog.response_body_text.like(like),
                        RequestLog.completion_text.like(like),
                    )
                )
                .order_by(desc(RequestLog.timestamp))
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = list(result.scalars().all())
        return LogListResponse(
            items=[_row_to_summary(r) for r in rows],
            next_cursor=None,
            has_more=False,
        )

    @router.get("/{log_id}", response_model=LogDetail)
    async def get_log(log_id: str) -> LogDetail:
        """Return one full log row including raw bodies and stream chunks."""
        sf = _require_session_factory()
        async with sf() as session:
            row = await session.get(RequestLog, log_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="log not found")
        return _row_to_detail(row)

    @router.delete("", response_model=DeleteResult)
    async def delete_logs(
        before: dt.datetime | None = Query(
            default=None, description="Delete rows older than this ISO timestamp."
        ),
        log_id: str | None = Query(default=None),
        all_: bool = Query(default=False, alias="all"),
    ) -> DeleteResult:
        """Delete rows by id, by ``before=`` cutoff, or ``all=true``.

        At least one of ``log_id`` / ``before`` / ``all`` must be set so
        a misclicked DELETE doesn't wipe history. ``all=true`` is an
        explicit "yes please trash everything" knob.
        """
        if before is None and log_id is None and not all_:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="specify before=, log_id= or all=true",
            )
        sf = _require_session_factory()
        async with sf() as session:
            stmt = delete(RequestLog)
            if log_id is not None:
                stmt = stmt.where(RequestLog.id == log_id)
            elif before is not None:
                stmt = stmt.where(RequestLog.timestamp < before)
            # all=true matches no extra WHERE, so the bare DELETE wipes.
            result = await session.execute(stmt)
            await session.commit()
        return DeleteResult(deleted=int(result.rowcount or 0))

    return router


__all__ = [
    "DeleteResult",
    "LogDetail",
    "LogListResponse",
    "LogStats",
    "LogSummary",
    "make_logs_router",
]
