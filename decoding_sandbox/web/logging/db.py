"""SQLAlchemy async engine + ORM for the upstream-request log store.

The store is a single SQLite file (``~/.local/share/dsbx/logs.db`` by
default; overridable via ``[web.logging].db_path`` in ``config.toml``).
SQLite is the right tool here: the dsbx web middleware is single-user
single-process, the write rate is human-scale (a few rows per minute at
most), and the on-disk format makes "blow away my history" a one-liner
(``rm logs.db``).

This module is the analogue of ``a production proxy gateway/backend/ai_proxy/db/engine.py``
plus the ``ProxyRequest`` half of ``ai_proxy/db/models.py``, trimmed for
SQLite (no JSONB, no TSVECTOR, no GIN -- ``JSON``-as-text columns and a
handful of plain b-tree indexes are enough at our scale).
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

log = logging.getLogger("decoding_sandbox.web.logging.db")


# Module-level singletons. The lifespan hook in ``web/app.py`` initializes
# them on startup and disposes on shutdown; tests call init_engine with an
# in-memory URL and bypass the lifespan entirely.
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for the log store."""


class RequestLog(Base):
    """One upstream HTTP call.

    ``id`` is a 36-char UUID string (SQLite has no native UUID type, and
    ``String(36)`` keeps the column human-readable in a SQLite browser).
    ``timestamp`` is indexed because every list query orders by it
    descending. The four ``*_body`` columns are ``JSON``, which SQLAlchemy
    serializes/deserializes through the SQLite ``json1`` extension; the
    matching ``*_body_text`` columns hold the same content as a flat
    string so ``LIKE`` search stays cheap (no JSON probe per row).
    """

    __tablename__ = "request_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # Which backend made this call.
    backend_name: Mapped[str] = mapped_column(String(128), index=True, default="")
    backend_family: Mapped[str] = mapped_column(String(32), default="")
    provider_name: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    # Outgoing request.
    method: Mapped[str] = mapped_column(String(10), default="POST")
    upstream_url: Mapped[str] = mapped_column(String(2048), default="")
    upstream_path: Mapped[str] = mapped_column(String(512), default="", index=True)
    request_headers: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    request_body: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    request_body_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Upstream response.
    response_status_code: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    response_headers: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    response_body: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    response_body_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    stream_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    is_streaming: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timing.
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Token usage (when the upstream reported a usage block OR the SSE
    # merger reconstructed it from the stream's terminating ``usage``
    # frame).
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Denormalized helpers the UI list page reads cheaply.
    model_resolved: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    completion_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_request_logs_backend_timestamp", "backend_name", "timestamp"),
    )


def _normalize_db_url(db_path_or_url: str) -> str:
    """Accept either an ``sqlite+aiosqlite://`` URL or a bare path.

    For ergonomics: ``[web.logging].db_path = "~/.local/share/dsbx/logs.db"``
    should "just work" without users having to know the SQLAlchemy URL
    incantation. We also expand ``~`` and environment variables; the
    parent directory is created on demand so a fresh deployment doesn't
    crash on first write.
    """
    if "://" in db_path_or_url:
        return db_path_or_url
    expanded = Path(os.path.expandvars(os.path.expanduser(db_path_or_url)))
    expanded.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{expanded}"


async def init_engine(db_path_or_url: str) -> AsyncEngine:
    """Create the async engine, run ``Base.metadata.create_all``, return engine.

    Idempotent: calling twice with the same URL is fine -- the second
    call disposes the previous engine first. ``create_all`` is cheap and
    only adds missing tables, so we don't need a migration system at our
    scale; if/when the schema changes incompatibly, the operator deletes
    ``logs.db`` and starts fresh.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None

    url = _normalize_db_url(db_path_or_url)
    log.info("dsbx-web: opening log store at %s", url)
    # ``connect_args={"timeout": 30}`` lets a writer wait 30 s for a
    # concurrent read to release the SQLite lock instead of failing
    # immediately. The flush task batches writes so contention is rare,
    # but the API read path runs on the same loop and can occasionally
    # land in the middle of a flush.
    _engine = create_async_engine(url, connect_args={"timeout": 30})
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    return _engine


async def dispose_engine() -> None:
    """Close the engine on shutdown; idempotent."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


def get_engine() -> AsyncEngine | None:
    """Return the current async engine (or ``None`` if not initialized)."""
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    """Return the current session factory (or ``None`` if not initialized)."""
    return _session_factory


__all__ = [
    "Base",
    "RequestLog",
    "dispose_engine",
    "get_engine",
    "get_session_factory",
    "init_engine",
]
