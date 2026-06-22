"""Tests for :class:`decoding_sandbox.web.logging.transport.LoggingTransport`.

We feed canned responses through ``httpx.MockTransport`` and drain the
asyncio queue to assert that exactly the right :class:`LogEntry` shapes
land. The transport itself is fully synchronous; the queue lives on
the asyncio loop, so each test creates one queue inline and runs the
transport call inside ``asyncio.run`` (with the transport's loop set to
the running loop) so the bridge through ``run_coroutine_threadsafe``
exercises the real code path.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from decoding_sandbox.web.logging import service as logsvc
from decoding_sandbox.web.logging.aggregator import aggregate_stream
from decoding_sandbox.web.logging.models import LogEntry
from decoding_sandbox.web.logging.transport import LoggingTransport


# --------------------------------------------------------------------------- #
# Test helpers: small queue management without a running flush task
# --------------------------------------------------------------------------- #
@pytest.fixture
def queue_setup():
    """Install a fresh queue on the service module for the test.

    We bypass ``start_logging_service`` because we don't want the flush
    task running -- we'll inspect the queue contents directly.
    """
    logsvc._queue = asyncio.Queue(maxsize=10000)
    yield logsvc._queue
    logsvc._queue = None


async def _drain(queue: asyncio.Queue[LogEntry]) -> list[LogEntry]:
    """Pull everything currently on the queue."""
    # Give run_coroutine_threadsafe one tick to actually deliver.
    await asyncio.sleep(0.01)
    out: list[LogEntry] = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def _run_request(transport: LoggingTransport, request: httpx.Request, *, stream_consume=True):
    """Run a request through the transport and (for streamed responses) iterate it.

    Mirrors what an actual backend does: open the response, read body or
    iterate stream, close. We split the iteration step out so tests can
    simulate an early-close cancellation.
    """
    response = transport.handle_request(request)
    if stream_consume:
        body = b"".join(response.stream)
        return response, body
    return response, b""


# --------------------------------------------------------------------------- #
# Non-streaming JSON response
# --------------------------------------------------------------------------- #
async def test_non_streaming_json_logs_one_entry(queue_setup):
    queue = queue_setup
    body = json.dumps(
        {
            "id": "cmpl-123",
            "model": "accounts/fireworks/models/gpt-oss-120b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
        }
    ).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"Content-Type": "application/json"},
        )

    transport = LoggingTransport(
        inner=httpx.MockTransport(handler),
        backend_name="fireworks",
        backend_family="cloud",
        provider_name="fireworks",
        upstream_base_url="https://api.fireworks.ai/inference/v1",
        loop=asyncio.get_running_loop(),
    )

    request = httpx.Request(
        "POST",
        "https://api.fireworks.ai/inference/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer secret-key-1234567890"},
    )
    transport.handle_request(request)
    entries = await _drain(queue)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.backend_name == "fireworks"
    assert entry.provider_name == "fireworks"
    assert entry.method == "POST"
    assert entry.upstream_path == "/inference/v1/chat/completions"
    assert entry.response_status_code == 200
    assert not entry.is_streaming
    assert entry.prompt_tokens == 7
    assert entry.completion_tokens == 1
    assert entry.total_tokens == 8
    assert entry.model_resolved in {"x", "accounts/fireworks/models/gpt-oss-120b"}
    assert isinstance(entry.response_body, dict)
    assert entry.latency_ms is not None and entry.latency_ms >= 0
    assert entry.ttft_ms is None  # ttft only applies to streamed bodies
    # request body is captured; the model field is preserved in the JSON
    assert isinstance(entry.request_body, dict)
    assert entry.request_body.get("model") == "x"
    # Bearer header is captured in raw form (httpx lowercases header
    # keys); masking happens at write time (see test_logs_api.py for
    # the masked-write assertion).
    headers_lower = {k.lower(): v for k, v in (entry.request_headers or {}).items()}
    assert "authorization" in headers_lower
    assert "Bearer" in headers_lower["authorization"]


# --------------------------------------------------------------------------- #
# OpenAI SSE stream
# --------------------------------------------------------------------------- #
async def test_openai_sse_stream_merges_into_single_entry(queue_setup):
    queue = queue_setup
    frames = [
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}],"model":"gpt-oss"}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"lo"}}],"model":"gpt-oss"}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":3,"total_tokens":6}}\n\n',
        b"data: [DONE]\n\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b"".join(frames)),
            headers={"Content-Type": "text/event-stream"},
        )

    transport = LoggingTransport(
        inner=httpx.MockTransport(handler),
        backend_name="fireworks",
        backend_family="cloud",
        provider_name="fireworks",
        upstream_base_url="https://api.fireworks.ai/inference/v1",
        loop=asyncio.get_running_loop(),
    )
    request = httpx.Request(
        "POST",
        "https://api.fireworks.ai/inference/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    _run_request(transport, request)
    entries = await _drain(queue)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.is_streaming is True
    assert entry.response_status_code == 200
    assert entry.ttft_ms is not None
    body = entry.response_body
    assert isinstance(body, dict)
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert entry.completion_text == "Hello world"
    assert entry.stop_reason == "stop"
    assert entry.prompt_tokens == 3
    assert entry.completion_tokens == 3
    assert entry.total_tokens == 6
    # Stream chunks captured 1:1 (minus the [DONE] terminator).
    assert entry.stream_chunks is not None and len(entry.stream_chunks) == 3


# --------------------------------------------------------------------------- #
# dsbx-native SSE stream
# --------------------------------------------------------------------------- #
async def test_dsbx_sse_stream_merges_into_single_entry(queue_setup):
    queue = queue_setup
    frames = [
        b'data: {"event":"step","step":{"decision":{"token_text":"Hi"}}}\n\n',
        b'data: {"event":"step","step":{"decision":{"token_text":" there"}}}\n\n',
        b'data: {"event":"usage","prompt_tokens":2,"completion_tokens":2,"total_tokens":4}\n\n',
        b'data: {"event":"done","stop_reason":"eos"}\n\n',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b"".join(frames)),
            headers={"Content-Type": "text/event-stream"},
        )

    transport = LoggingTransport(
        inner=httpx.MockTransport(handler),
        backend_name="dsbx-host-py",
        backend_family="remote",
        provider_name=None,
        upstream_base_url="http://192.0.2.42:8000",
        loop=asyncio.get_running_loop(),
    )
    request = httpx.Request(
        "POST",
        "http://192.0.2.42:8000/v1/generate/stream",
        json={"prompt": "Hello"},
    )
    _run_request(transport, request)
    entries = await _drain(queue)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.is_streaming is True
    assert entry.backend_name == "dsbx-host-py"
    assert entry.prompt_tokens == 2
    assert entry.completion_tokens == 2
    assert entry.total_tokens == 4
    assert entry.stop_reason == "eos"
    assert entry.completion_text == "Hi there"
    # Stream chunks: 2 steps + 1 usage + 1 done = 4 frames retained.
    assert entry.stream_chunks is not None and len(entry.stream_chunks) == 4


# --------------------------------------------------------------------------- #
# Cancellation: caller closes the stream before exhausting it
# --------------------------------------------------------------------------- #
async def test_stream_close_before_completion_records_partial_entry(queue_setup):
    queue = queue_setup
    # Two valid frames, then we close the stream early.
    frames = [
        b'data: {"choices":[{"index":0,"delta":{"content":"par"}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"tial"}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":" never read"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b"".join(frames)),
            headers={"Content-Type": "text/event-stream"},
        )

    transport = LoggingTransport(
        inner=httpx.MockTransport(handler),
        backend_name="fireworks",
        backend_family="cloud",
        provider_name="fireworks",
        upstream_base_url="https://api.fireworks.ai/inference/v1",
        loop=asyncio.get_running_loop(),
    )
    request = httpx.Request(
        "POST",
        "https://api.fireworks.ai/inference/v1/chat/completions",
        json={"model": "x", "messages": []},
    )
    response = transport.handle_request(request)
    # Close the stream without iterating it. The transport's
    # TeeStream.close() path emits a log entry for the attempt
    # regardless of how many bytes were consumed.
    response.stream.close()

    entries = await _drain(queue)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.is_streaming is True
    assert entry.response_status_code == 200
    # We expect the partial-cancellation marker on the error_message
    # field whenever close() runs before iteration finishes.
    assert (entry.error_message or "").startswith("stream closed")


# --------------------------------------------------------------------------- #
# Aggregator unit tests (separately, no httpx in the loop)
# --------------------------------------------------------------------------- #
def test_aggregate_openai_stream_joins_deltas_and_extracts_usage():
    raw = (
        b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"b"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"c"}, "finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":3,"total_tokens":4}}\n\n'
        b"data: [DONE]\n\n"
    )
    agg = aggregate_stream(raw)
    assert agg.assembled_body is not None
    assert agg.assembled_body["choices"][0]["message"]["content"] == "abc"
    assert agg.assembled_body["choices"][0]["finish_reason"] == "length"
    assert agg.prompt_tokens == 1
    assert agg.completion_tokens == 3
    assert agg.total_tokens == 4
    assert agg.stop_reason == "length"
    assert agg.completion_text == "abc"


def test_aggregate_dsbx_stream_joins_step_tokens_and_captures_usage():
    raw = (
        b'data: {"event":"step","step":{"decision":{"token_text":"hello"}}}\n\n'
        b'data: {"event":"step","step":{"decision":{"token_text":" world"}}}\n\n'
        b'data: {"event":"usage","prompt_tokens":5,"completion_tokens":2,"total_tokens":7}\n\n'
        b'data: {"event":"done","stop_reason":"eos"}\n\n'
    )
    agg = aggregate_stream(raw)
    assert agg.completion_text == "hello world"
    assert agg.prompt_tokens == 5
    assert agg.completion_tokens == 2
    assert agg.total_tokens == 7
    assert agg.stop_reason == "eos"
    assert agg.assembled_body is not None
    assert agg.assembled_body["completion"] == "hello world"


def test_aggregate_empty_input_returns_empty_result():
    agg = aggregate_stream(b"")
    assert agg.assembled_body is None
    assert agg.chunks == []
    assert agg.completion_text is None
