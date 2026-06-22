"""``httpx.BaseTransport`` that captures every upstream call for logging.

Installed by :class:`decoding_sandbox.web.backends.BackendRegistry` when
building the three HTTP-backed backends (``RemoteBackend``,
``OpenAICompatBackend``, ``LlamaCppBackend``). The CLI path keeps using
httpx's default transport, so plain TUI installs (``pip install -e .``)
never need SQLAlchemy or aiosqlite installed.

Capture contract:

- For non-streaming responses: we ``read()`` the body up front, build a
  :class:`LogEntry`, enqueue it, then return the response unchanged (the
  caller's subsequent ``.json()`` re-uses the cached body, no extra
  read).
- For streaming responses: we replace ``response.stream`` with a tee
  iterator that captures every byte as the backend iterates it. We
  measure TTFT from request-issued to the first non-empty chunk, and
  emit the log entry when the iterator is exhausted (or closed early).
  This works for both ``response.iter_bytes()`` and the lower-level
  ``response.stream`` access -- both internally consume the same
  underlying ``ByteStream``.

Threading note: ``httpx.Client`` is synchronous, and the dsbx web layer
runs it inside Starlette's threadpool (so a blocking forward pass
doesn't stall the event loop). Our enqueue path therefore lives in a
worker thread; we bridge to the event loop with
``asyncio.run_coroutine_threadsafe`` using the loop reference captured
at transport construction time. The event loop is the one created by
``uvicorn`` for the FastAPI app, captured by ``make_web_app``'s lifespan
hook BEFORE the registry hands out any backends.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
from httpx import AsyncByteStream, SyncByteStream

from decoding_sandbox.web.logging.aggregator import aggregate_stream
from decoding_sandbox.web.logging.models import LogEntry
from decoding_sandbox.web.logging.service import enqueue_log

log = logging.getLogger("decoding_sandbox.web.logging.transport")

# Limit on how many bytes we keep of a non-streaming response body before
# truncating. Most logprob/inspect responses are <100 kB; bumping this
# to 4 MiB covers a very fat catalogue fetch without ever risking memory
# pressure from a stuck stream that returns multi-gigabyte garbage.
_MAX_BODY_BYTES = 4 * 1024 * 1024


class LoggingTransport(httpx.BaseTransport):
    """Wraps another :class:`httpx.BaseTransport` to capture + enqueue logs.

    Construction parameters:

    - ``inner``: the real transport to delegate to. Defaults to a fresh
      :class:`httpx.HTTPTransport`. Tests pass in :class:`httpx.MockTransport`
      so they can feed canned responses without touching the network.
    - ``backend_name`` / ``backend_family`` / ``provider_name``: identity
      fields stamped onto every entry this transport emits. The registry
      builds one transport per backend so all calls from a given backend
      get tagged consistently.
    - ``upstream_base_url``: stored on the log row for the UI's
      "where did this go?" column. We DO log it -- in a production proxy gateway we'd
      strip it for the browser, but here logs are operator-only behind
      the bearer-token gate, so showing the full URL is more useful
      than hiding it.
    - ``loop``: the event loop to schedule enqueues on. Captured at
      construction time so the synchronous transport call site doesn't
      have to ``asyncio.get_event_loop()`` (which would fail on a
      worker thread).
    """

    def __init__(
        self,
        *,
        inner: httpx.BaseTransport | None = None,
        backend_name: str = "",
        backend_family: str = "",
        provider_name: str | None = None,
        upstream_base_url: str = "",
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._inner = inner or httpx.HTTPTransport()
        self._backend_name = backend_name
        self._backend_family = backend_family
        self._provider_name = provider_name
        self._upstream_base_url = upstream_base_url
        # Capture the loop now; the request will run on a worker thread
        # where ``asyncio.get_event_loop()`` does not return the FastAPI
        # loop. None is allowed for tests that explicitly drive the
        # queue synchronously.
        self._loop = loop

    # ------------------------------------------------------------------ #
    # httpx.BaseTransport contract
    # ------------------------------------------------------------------ #
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        request_capture = _capture_request(request)
        request_id = uuid.uuid4()

        try:
            response = self._inner.handle_request(request)
        except httpx.RequestError as exc:
            self._enqueue_error(request_id, request_capture, exc, started)
            raise

        # We need a content-type and headers regardless of streaming or
        # not; snapshot them now while we still have a guaranteed-fresh
        # response object.
        response_headers = _headers_to_dict(response.headers)
        content_type = (response_headers.get("content-type") or "").lower()
        is_streaming = _looks_streaming(response, content_type)

        if not is_streaming:
            # Read+restore the body so the backend's subsequent
            # ``r.read()`` / ``r.json()`` doesn't re-trigger a network
            # read (httpx caches once ``.read()`` runs).
            try:
                body = response.read()
            except httpx.HTTPError as exc:
                self._enqueue_error(request_id, request_capture, exc, started)
                raise
            self._enqueue_completed(
                request_id=request_id,
                request_capture=request_capture,
                response=response,
                response_headers=response_headers,
                started=started,
                streaming=False,
                body=body,
            )
            return response

        # Streaming path: tee the bytes.
        original_stream = response.stream
        captured: list[bytes] = []
        ttft_holder: dict[str, float | None] = {"ms": None}
        wrapper = _TeeStream(
            inner=original_stream,
            captured=captured,
            ttft_holder=ttft_holder,
            started=started,
            on_done=lambda error: self._enqueue_completed(
                request_id=request_id,
                request_capture=request_capture,
                response=response,
                response_headers=response_headers,
                started=started,
                streaming=True,
                body=b"".join(captured),
                ttft_ms=ttft_holder["ms"],
                stream_error=error,
            ),
        )
        response.stream = wrapper  # type: ignore[assignment]
        return response

    # ------------------------------------------------------------------ #
    # internal: build + enqueue
    # ------------------------------------------------------------------ #
    def _enqueue_completed(
        self,
        *,
        request_id: uuid.UUID,
        request_capture: "_RequestCapture",
        response: httpx.Response,
        response_headers: dict[str, str],
        started: float,
        streaming: bool,
        body: bytes,
        ttft_ms: float | None = None,
        stream_error: str | None = None,
    ) -> None:
        latency_ms = (time.monotonic() - started) * 1000.0
        entry = LogEntry(
            id=request_id,
            backend_name=self._backend_name,
            backend_family=self._backend_family,
            provider_name=self._provider_name,
            method=request_capture.method,
            upstream_url=request_capture.url,
            upstream_path=request_capture.path,
            request_headers=request_capture.headers,
            request_body=request_capture.body_json,
            request_body_raw=request_capture.body_text,
            response_status_code=response.status_code,
            response_headers=response_headers,
            is_streaming=streaming,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            model_resolved=request_capture.model_resolved,
        )

        if streaming:
            aggregated = aggregate_stream(body)
            entry.response_body = aggregated.assembled_body
            entry.response_body_raw = _safe_decode(body)
            entry.stream_chunks = aggregated.chunks or None
            entry.prompt_tokens = aggregated.prompt_tokens
            entry.completion_tokens = aggregated.completion_tokens
            entry.total_tokens = aggregated.total_tokens
            entry.completion_text = aggregated.completion_text
            entry.stop_reason = aggregated.stop_reason
            entry.model_resolved = entry.model_resolved or aggregated.model_resolved
            entry.error_message = aggregated.error_message or stream_error
        else:
            parsed = _try_parse_json(body)
            entry.response_body = parsed
            entry.response_body_raw = _safe_decode(body)
            # Some upstream errors are JSON like ``{"error": {...}}`` or
            # ``{"detail": "..."}``; surface a short string into
            # error_message so the row glance can flag it red.
            entry.error_message = _extract_error_message(parsed, response.status_code)
            entry.prompt_tokens, entry.completion_tokens, entry.total_tokens = _extract_usage(parsed)
            # Many providers echo the resolved model back; harvest if we
            # didn't already pick it off the request.
            if entry.model_resolved is None:
                entry.model_resolved = _extract_model_resolved(parsed)

        self._dispatch(entry)

    def _enqueue_error(
        self,
        request_id: uuid.UUID,
        request_capture: "_RequestCapture",
        exc: Exception,
        started: float,
    ) -> None:
        latency_ms = (time.monotonic() - started) * 1000.0
        entry = LogEntry(
            id=request_id,
            backend_name=self._backend_name,
            backend_family=self._backend_family,
            provider_name=self._provider_name,
            method=request_capture.method,
            upstream_url=request_capture.url,
            upstream_path=request_capture.path,
            request_headers=request_capture.headers,
            request_body=request_capture.body_json,
            request_body_raw=request_capture.body_text,
            response_status_code=None,
            latency_ms=latency_ms,
            model_resolved=request_capture.model_resolved,
            error_message=f"{type(exc).__name__}: {exc}",
        )
        self._dispatch(entry)

    def _dispatch(self, entry: LogEntry) -> None:
        """Bridge the entry onto the event loop (or run synchronously in tests)."""
        if self._loop is None:
            # No loop captured -- typically a unit test driving
            # handle_request directly without a running FastAPI. Run
            # the coroutine on a private loop so the queue still gets
            # populated. The asyncio module accepts this pattern via
            # ``asyncio.run``.
            try:
                asyncio.run(enqueue_log(entry))
            except RuntimeError:
                # If we're somehow inside a running loop (test mishap),
                # fall back to dropping a structured warning rather
                # than re-raising and aborting the upstream call.
                log.warning("dsbx-web: cannot enqueue log without a loop")
            return
        try:
            asyncio.run_coroutine_threadsafe(enqueue_log(entry), self._loop)
        except RuntimeError:
            log.warning("dsbx-web: failed to schedule log entry (loop stopped?)")


# --------------------------------------------------------------------------- #
# Captured request snapshot
# --------------------------------------------------------------------------- #
class _RequestCapture:
    """Plain attrs container so we can pass one thing around."""

    __slots__ = ("method", "url", "path", "headers", "body_json", "body_text", "model_resolved")

    def __init__(
        self,
        method: str,
        url: str,
        path: str,
        headers: dict[str, str],
        body_json: Any,
        body_text: str | None,
        model_resolved: str | None,
    ) -> None:
        self.method = method
        self.url = url
        self.path = path
        self.headers = headers
        self.body_json = body_json
        self.body_text = body_text
        self.model_resolved = model_resolved


def _capture_request(request: httpx.Request) -> _RequestCapture:
    """Snapshot the outgoing request shape (headers + body) for the log row.

    The body bytes are read out of the request once. httpx allows reading
    ``request.content`` repeatedly on a "memory" stream (which is the
    default for the JSON / dict bodies the backends send), so this
    doesn't disturb the actual upload.
    """
    try:
        body_bytes = request.content
    except Exception:  # noqa: BLE001 -- defensive against custom streams
        body_bytes = b""
    body_text: str | None = None
    body_json: Any = None
    if body_bytes:
        body_text = _safe_decode(body_bytes)
        body_json = _try_parse_json(body_bytes)

    model_resolved = None
    if isinstance(body_json, dict):
        m = body_json.get("model")
        if isinstance(m, str):
            model_resolved = m

    return _RequestCapture(
        method=request.method,
        url=str(request.url),
        path=request.url.path or "",
        headers=_headers_to_dict(request.headers),
        body_json=body_json,
        body_text=body_text,
        model_resolved=model_resolved,
    )


# --------------------------------------------------------------------------- #
# Tee stream wrapper
# --------------------------------------------------------------------------- #
class _TeeStream(SyncByteStream, AsyncByteStream):
    """Mimics the httpx ``ByteStream`` protocol while teeing into a buffer.

    httpx iterates ``response.stream`` via ``__iter__`` (sync) and
    ``__aiter__`` (async); ``httpx._client._send_single_request`` asserts
    the response stream isinstance(SyncByteStream) when using
    ``httpx.Client``, so this class explicitly inherits from BOTH the
    sync and async ByteStream protocols. The sync backends we
    instrument today only use the sync path, but having the async one
    means an async backend added later picks up logging for free.

    ``on_done`` is invoked exactly once when the stream finishes or is
    closed, with an optional error string -- ``None`` for a clean
    end-of-stream, a short message string for a mid-stream exception
    or close.
    """

    __slots__ = ("_inner", "_captured", "_ttft_holder", "_started", "_on_done", "_emitted")

    def __init__(
        self,
        *,
        inner: Any,
        captured: list[bytes],
        ttft_holder: dict[str, float | None],
        started: float,
        on_done,
    ) -> None:
        self._inner = inner
        self._captured = captured
        self._ttft_holder = ttft_holder
        self._started = started
        self._on_done = on_done
        self._emitted = False  # guard so we only emit one log per stream

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._inner:
                if chunk:
                    if self._ttft_holder["ms"] is None:
                        self._ttft_holder["ms"] = (time.monotonic() - self._started) * 1000.0
                    self._captured.append(chunk)
                yield chunk
        except Exception as exc:  # noqa: BLE001
            self._finish(error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self._finish(error=None)

    async def __aiter__(self):
        try:
            async for chunk in self._inner:  # type: ignore[union-attr]
                if chunk:
                    if self._ttft_holder["ms"] is None:
                        self._ttft_holder["ms"] = (time.monotonic() - self._started) * 1000.0
                    self._captured.append(chunk)
                yield chunk
        except Exception as exc:  # noqa: BLE001
            self._finish(error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self._finish(error=None)

    def close(self) -> None:
        # Called by httpx when the response context manager exits.
        # If the iterator never finished (e.g. caller aborted), record a
        # partial entry so the log still has a row for the attempt.
        if not self._emitted:
            self._finish(error="stream closed before completion")
        inner_close = getattr(self._inner, "close", None)
        if callable(inner_close):
            inner_close()

    async def aclose(self) -> None:
        if not self._emitted:
            self._finish(error="stream closed before completion")
        inner_aclose = getattr(self._inner, "aclose", None)
        if callable(inner_aclose):
            await inner_aclose()

    def _finish(self, *, error: str | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        try:
            self._on_done(error)
        except Exception:  # noqa: BLE001 -- never let logging break the caller
            log.exception("dsbx-web: log entry emission failed for stream")


# --------------------------------------------------------------------------- #
# Helpers (header / body parsing, error / usage extraction)
# --------------------------------------------------------------------------- #
def _headers_to_dict(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for k, v in headers.items():
            out[str(k)] = str(v)
    except Exception:  # noqa: BLE001
        return {}
    return out


def _looks_streaming(response: httpx.Response, content_type: str) -> bool:
    if "text/event-stream" in content_type:
        return True
    if response.headers.get("transfer-encoding", "").lower() == "chunked":
        # Some llama.cpp builds use chunked transfer for non-SSE JSON;
        # tee them only if the content_type is also event-stream (the
        # branch above), otherwise fall through to the non-streaming
        # path so we get a parsed JSON body in the log.
        return False
    return False


def _safe_decode(body: bytes) -> str | None:
    if not body:
        return None
    truncated = body[:_MAX_BODY_BYTES]
    try:
        return truncated.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None


def _try_parse_json(body: bytes) -> Any:
    if not body:
        return None
    import json

    try:
        return json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeError):
        return None


def _extract_error_message(parsed: Any, status_code: int | None) -> str | None:
    if status_code is not None and status_code < 400:
        return None
    if not isinstance(parsed, dict):
        return None
    # FastAPI-style ``{"detail": "..."}``.
    detail = parsed.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    # OpenAI / OpenRouter style ``{"error": {"message": "..."}}``.
    error_obj = parsed.get("error")
    if isinstance(error_obj, dict):
        msg = error_obj.get("message")
        if isinstance(msg, str):
            return msg
    if isinstance(error_obj, str) and error_obj:
        return error_obj
    return None


def _extract_usage(parsed: Any) -> tuple[int | None, int | None, int | None]:
    if not isinstance(parsed, dict):
        return None, None, None
    usage = parsed.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    def _coerce(v: Any) -> int | None:
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        if isinstance(v, float):
            return int(v)
        return None
    return (
        _coerce(usage.get("prompt_tokens")),
        _coerce(usage.get("completion_tokens")),
        _coerce(usage.get("total_tokens")),
    )


def _extract_model_resolved(parsed: Any) -> str | None:
    if not isinstance(parsed, dict):
        return None
    m = parsed.get("model")
    if isinstance(m, str):
        return m
    return None


__all__ = ["LoggingTransport"]
