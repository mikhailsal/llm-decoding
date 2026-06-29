"""Upstream-request logging for the dsbx web middleware.

When the middleware fronts an HTTP-backed backend (``RemoteBackend`` against
dsbx-host, ``OpenAICompatBackend`` against a cloud provider, ``LlamaCppBackend``
against a local ``llama-server``), every outgoing call is captured at the
:class:`httpx.BaseTransport` boundary by :class:`LoggingTransport` and
enqueued onto an asyncio queue. A background task drains the queue into a
SQLite database via SQLAlchemy. Streamed responses are tee'd and merged so
each upstream call lands as a single fat row, mirroring the design in
a production proxy gateway.

Public surface:

- :class:`LogEntry`       -- the in-memory record built by the transport.
- :class:`LoggingTransport` -- the httpx hook the BackendRegistry installs.
- :func:`start_logging_service` / :func:`stop_logging_service` -- lifespan
  hooks for the FastAPI app.
- :func:`init_engine` / :func:`dispose_engine` -- SQLite engine bootstrap.

Nothing from this package is imported on the CLI path; ``httpx.Client``
without an explicit ``transport=`` argument keeps using the default httpx
HTTPTransport, so the TUI install stays free of SQLAlchemy / aiosqlite.
"""

from __future__ import annotations

from decoding_sandbox.web.logging.db import (
    RequestLog,
    dispose_engine,
    get_engine,
    get_session_factory,
    init_engine,
)
from decoding_sandbox.web.logging.models import LogEntry
from decoding_sandbox.web.logging.service import (
    enqueue_log,
    start_logging_service,
    stop_logging_service,
)
from decoding_sandbox.web.logging.transport import LoggingTransport

__all__ = [
    "LogEntry",
    "LoggingTransport",
    "RequestLog",
    "dispose_engine",
    "enqueue_log",
    "get_engine",
    "get_session_factory",
    "init_engine",
    "start_logging_service",
    "stop_logging_service",
]
