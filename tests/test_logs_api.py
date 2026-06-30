"""Tests for ``dsbx.web.logs_api`` against a seeded SQLite DB.

We bring up an in-memory SQLite engine, seed a handful of ``RequestLog``
rows directly, then drive the FastAPI router through ``TestClient`` to
make sure the list / detail / search / stats / delete endpoints all
return what they should. We do NOT go through the LoggingTransport here
-- that's covered in :mod:`tests.test_logging_transport` and we want a
clean focus on the API layer.
"""

from __future__ import annotations

import asyncio
import contextlib
import os as _os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dsbx.web.logging import db as logdb
from dsbx.web.logging.db import RequestLog
from dsbx.web.logs_api import make_logs_router


# --------------------------------------------------------------------------- #
# Fixture: app with seeded in-memory DB
# --------------------------------------------------------------------------- #
@pytest.fixture
def seeded_app():
    """Seed three rows into a private SQLite file DB and build a FastAPI app.

    We use a private temp file path rather than ``:memory:`` because
    SQLAlchemy's async engine treats every connection to ``:memory:`` as
    a fresh database -- the schema we create_all on connection #1 would
    be missing on connection #2 of the same engine. A file-backed DB
    sidesteps that without paying meaningfully for it at test scale.
    """
    fd, db_path = tempfile.mkstemp(prefix="dsbx-test-logs-", suffix=".db")
    _os.close(fd)
    loop = asyncio.new_event_loop()

    async def setup():
        await logdb.init_engine(db_path)
        session_factory = logdb.get_session_factory()
        assert session_factory is not None
        now = datetime.now(tz=timezone.utc)
        async with session_factory() as session:
            session.add(
                RequestLog(
                    id=str(uuid.uuid4()),
                    timestamp=now - timedelta(minutes=2),
                    backend_name="dsbx-host-py",
                    backend_family="remote",
                    method="POST",
                    upstream_url="http://192.0.2.42:8000/v1/generate/stream",
                    upstream_path="/v1/generate/stream",
                    request_headers={"Authorization": "Bearer secret-key-1234567890"},
                    request_body={"prompt": "Hello", "max_tokens": 4},
                    request_body_text='{"prompt": "Hello"}',
                    response_status_code=200,
                    response_body={"completion": "Hi there"},
                    response_body_text='{"completion": "Hi there"}',
                    is_streaming=True,
                    latency_ms=420.0,
                    ttft_ms=120.0,
                    prompt_tokens=2,
                    completion_tokens=4,
                    total_tokens=6,
                    completion_text="Hi there",
                    stop_reason="eos",
                )
            )
            session.add(
                RequestLog(
                    id=str(uuid.uuid4()),
                    timestamp=now - timedelta(minutes=1),
                    backend_name="fireworks",
                    backend_family="cloud",
                    provider_name="fireworks",
                    method="POST",
                    upstream_url="https://api.fireworks.ai/inference/v1/chat/completions",
                    upstream_path="/inference/v1/chat/completions",
                    response_status_code=200,
                    response_body={"choices": [{"message": {"content": "ok"}}]},
                    is_streaming=False,
                    latency_ms=180.0,
                    prompt_tokens=3,
                    completion_tokens=1,
                    total_tokens=4,
                    model_resolved="accounts/fireworks/models/gpt-oss-120b",
                    completion_text="ok",
                )
            )
            session.add(
                RequestLog(
                    id=str(uuid.uuid4()),
                    timestamp=now,
                    backend_name="fireworks",
                    backend_family="cloud",
                    provider_name="fireworks",
                    method="POST",
                    upstream_url="https://api.fireworks.ai/inference/v1/chat/completions",
                    upstream_path="/inference/v1/chat/completions",
                    response_status_code=429,
                    response_body={"error": {"message": "rate limited"}},
                    is_streaming=False,
                    latency_ms=50.0,
                    error_message="rate limited",
                )
            )
            await session.commit()

    loop.run_until_complete(setup())

    # Auth dependency is a no-op for these tests (we already covered
    # auth in test_web_auth.py); the router still requires it for
    # registration shape, so we hand it a callable that does nothing.
    def fake_require_bearer():
        return None

    app = FastAPI()
    app.include_router(make_logs_router(fake_require_bearer))
    client = TestClient(app)
    try:
        yield client
    finally:

        async def teardown():
            await logdb.dispose_engine()

        loop.run_until_complete(teardown())
        loop.close()
        with contextlib.suppress(OSError):
            _os.unlink(db_path)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_list_returns_all_three_rows_newest_first(seeded_app):
    r = seeded_app.get("/api/v1/logs")
    assert r.status_code == 200
    body = r.json()
    items = body["items"]
    assert len(items) == 3
    # Newest first: errors (now) -> fireworks (now-1) -> dsbx-host-py (now-2).
    assert items[0]["error_message"] == "rate limited"
    assert items[2]["backend_name"] == "dsbx-host-py"


def test_list_filters_by_backend(seeded_app):
    r = seeded_app.get("/api/v1/logs?backend=dsbx-host-py")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["backend_name"] == "dsbx-host-py"


def test_list_filters_errors_only(seeded_app):
    r = seeded_app.get("/api/v1/logs?is_error=true")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["error_message"] == "rate limited"
    assert items[0]["response_status_code"] == 429


def test_list_cursor_pagination(seeded_app):
    r1 = seeded_app.get("/api/v1/logs?limit=1")
    body1 = r1.json()
    assert len(body1["items"]) == 1
    assert body1["has_more"] is True
    cursor = body1["next_cursor"]
    assert cursor is not None

    r2 = seeded_app.get(f"/api/v1/logs?limit=1&cursor={cursor}")
    body2 = r2.json()
    assert len(body2["items"]) == 1
    # Strictly older than the first page's last row.
    assert body2["items"][0]["id"] != body1["items"][0]["id"]


def test_get_log_returns_full_row(seeded_app):
    r = seeded_app.get("/api/v1/logs?backend=dsbx-host-py")
    log_id = r.json()["items"][0]["id"]
    r = seeded_app.get(f"/api/v1/logs/{log_id}")
    assert r.status_code == 200
    detail = r.json()
    # The detail shape carries the full request/response bodies the
    # summary shape didn't, plus the raw body text columns the search
    # endpoint searches over.
    assert detail["request_body"] == {"prompt": "Hello", "max_tokens": 4}
    assert detail["request_body_text"] == '{"prompt": "Hello"}'
    assert detail["response_body"] == {"completion": "Hi there"}
    assert detail["upstream_url"] == "http://192.0.2.42:8000/v1/generate/stream"


def test_get_log_missing_returns_404(seeded_app):
    r = seeded_app.get("/api/v1/logs/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_search_matches_body_text(seeded_app):
    # "Hello" appears in dsbx-host-py's request_body_text only.
    r = seeded_app.get("/api/v1/logs/search?q=Hello")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["backend_name"] == "dsbx-host-py"


def test_search_matches_error_message(seeded_app):
    # "limited" is only present in the error_message of the 429 row;
    # using "rate" would also match the dsbx-host-py path which contains
    # "/v1/generate/" -- the substring "rate" is the wrong query for
    # asserting error-only behavior.
    r = seeded_app.get("/api/v1/logs/search?q=limited")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["error_message"] == "rate limited"


def test_stats_returns_correct_counters(seeded_app):
    r = seeded_app.get("/api/v1/logs/stats")
    assert r.status_code == 200
    stats = r.json()
    assert stats["total"] == 3
    assert stats["streaming"] == 1
    assert stats["non_streaming"] == 2
    assert stats["error_count"] == 1
    assert stats["total_prompt_tokens"] == 5  # 2 + 3 + 0
    assert stats["total_completion_tokens"] == 5  # 4 + 1 + 0
    assert stats["avg_latency_ms"] is not None


def test_delete_by_id_removes_one_row(seeded_app):
    items = seeded_app.get("/api/v1/logs").json()["items"]
    target = items[0]["id"]
    r = seeded_app.delete(f"/api/v1/logs?log_id={target}")
    assert r.status_code == 200
    assert r.json()["deleted"] == 1
    remaining = seeded_app.get("/api/v1/logs").json()["items"]
    assert len(remaining) == 2
    assert all(item["id"] != target for item in remaining)


def test_delete_without_args_returns_400(seeded_app):
    r = seeded_app.delete("/api/v1/logs")
    assert r.status_code == 400


def test_delete_all_wipes_history(seeded_app):
    r = seeded_app.delete("/api/v1/logs?all=true")
    assert r.status_code == 200
    assert r.json()["deleted"] == 3
    assert seeded_app.get("/api/v1/logs").json()["items"] == []
