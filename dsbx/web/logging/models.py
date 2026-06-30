"""In-memory :class:`LogEntry` Pydantic model.

A :class:`LogEntry` is built by :class:`LoggingTransport` after each
upstream HTTP call completes (or fails) and is dropped onto an
``asyncio.Queue``. The background flush task in
:mod:`dsbx.web.logging.service` turns each entry into a row
in the ``RequestLog`` SQLAlchemy table.

The schema is intentionally trimmed compared to a production proxy gateway's ``LogEntry``
(which carries both ``client_*`` and provider-side doublets because the
proxy sits between two parties): the dsbx middleware only logs the
provider-side, so we keep ONE set of request / response fields. We also
add ``backend_name`` so the ``/logs`` UI can tell upstream calls apart
across the three families of HTTP backend the middleware talks to.

All fields default to ``None`` / sensible empties so the transport can
build a partial entry on the error path (e.g. a connect timeout) without
having to populate every column.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    """One upstream HTTP call captured by :class:`LoggingTransport`.

    Notes on the fields the UI cares about most:

    - ``request_body`` is the JSON we sent upstream (already masked for
      secrets); ``request_body_raw`` is the unparsed bytes (decoded
      utf-8 / errors=replace) so the detail view can show what actually
      went on the wire even when the body isn't JSON.
    - ``response_body`` is the (parsed) JSON of a non-streaming response
      OR the *assembled* JSON we synthesize from a merged SSE stream;
      ``stream_chunks`` holds every parsed SSE frame in order so the
      operator can step through one chunk at a time when debugging a
      glitchy provider.
    - ``model_resolved`` is the model name we actually sent upstream
      (e.g. ``"accounts/fireworks/models/gpt-oss-120b"``). For the
      RemoteBackend path where the dsbx-host server picks the model, this
      stays None.
    - ``error_message`` is set on transport-level failures (connect /
      timeout / mid-stream error) AND on upstream-reported 4xx/5xx with
      a parseable ``{"detail": ...}`` body.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Which backend the middleware was acting on behalf of when this call
    # went out. ``backend_name`` is the registry key (``dsbx-host-py`` /
    # ``fireworks`` / ``llamacpp``...); ``provider_name`` is the
    # OpenAI-compat provider when family=cloud (else None).
    backend_name: str = ""
    backend_family: str = ""  # "remote" | "cloud" | "local"
    provider_name: str | None = None

    # Outgoing request shape.
    method: str = "POST"
    upstream_url: str = ""
    upstream_path: str = ""
    request_headers: dict[str, Any] | None = None
    request_body: Any = None
    request_body_raw: str | None = None

    # Upstream response shape.
    response_status_code: int | None = None
    response_headers: dict[str, Any] | None = None
    response_body: Any = None
    response_body_raw: str | None = None
    stream_chunks: list[Any] | None = None
    is_streaming: bool = False

    # Timing.
    latency_ms: float | None = None
    ttft_ms: float | None = None

    # Token accounting (parsed out of the assembled response when present).
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # Convenience denormalizations the UI uses for the list page; cheap to
    # compute from the body and trivially searchable.
    model_resolved: str | None = None
    completion_text: str | None = None
    stop_reason: str | None = None

    error_message: str | None = None


__all__ = ["LogEntry"]
