import asyncio
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from dsbx.web.logging import db as logdb
from dsbx.web.logging import service as logsvc
from dsbx.web.logging.models import LogEntry


@pytest.fixture(autouse=True)
async def cleanup_service():
    """Ensure logging service state is clean before and after each test."""
    await logsvc.stop_logging_service()
    await logdb.dispose_engine()
    logsvc._queue = None
    logsvc._flush_task = None
    yield
    await logsvc.stop_logging_service()
    await logdb.dispose_engine()
    logsvc._queue = None
    logsvc._flush_task = None


@pytest.fixture
async def temp_db():
    fd, db_path = tempfile.mkstemp(prefix="dsbx-test-service-", suffix=".db")
    os.close(fd)
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


def make_test_log_entry() -> LogEntry:
    return LogEntry(
        id="c1234567-89ab-cdef-0123-456789abcdef",
        timestamp=datetime.now(tz=timezone.utc),
        backend_name="test-backend",
        backend_family="test-family",
        provider_name="test-provider",
        method="POST",
        upstream_url="http://example.com/v1/generate",
        upstream_path="/v1/generate",
        request_headers={"Authorization": "Bearer secret-key-123"},
        request_body={"prompt": "hi"},
        request_body_raw='{"prompt": "hi"}',
        response_status_code=200,
        response_headers={"Content-Type": "application/json"},
        response_body={"completion": "hello"},
        response_body_raw='{"completion": "hello"}',
        stream_chunks=None,
        is_streaming=False,
        latency_ms=10.0,
        ttft_ms=None,
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
        model_resolved="test-model",
        completion_text="hello",
        stop_reason="stop",
        error_message=None,
    )


async def test_get_queue_returns_none_when_inactive():
    assert logsvc.get_queue() is None


async def test_start_stop_service_idempotent(temp_db):
    await logdb.init_engine(temp_db)

    task1 = logsvc.start_logging_service(batch_size=10, flush_interval=0.1)
    await asyncio.sleep(0.01)  # let the loop run
    assert task1 is not None
    assert not task1.done()
    assert logsvc.get_queue() is not None

    task2 = logsvc.start_logging_service(batch_size=10, flush_interval=0.1)
    assert task2 is task1

    await logsvc.stop_logging_service()
    assert logsvc.get_queue() is None
    assert logsvc._flush_task is None


async def test_enqueue_log_no_op_when_inactive():
    entry = make_test_log_entry()
    # Should not raise any error
    await logsvc.enqueue_log(entry)


async def test_enqueue_log_queue_full(temp_db):
    await logdb.init_engine(temp_db)
    # Start with maxsize=1 queue
    logsvc._queue = asyncio.Queue(maxsize=1)

    entry = make_test_log_entry()
    await logsvc.enqueue_log(entry)

    # Second enqueue should be dropped and log a warning
    with patch("dsbx.web.logging.service.log") as mock_log:
        await logsvc.enqueue_log(entry)
        mock_log.warning.assert_called_once()


async def test_drain_for_test_no_queue_or_engine():
    assert await logsvc.drain_for_test() == 0

    logsvc._queue = asyncio.Queue()
    assert await logsvc.drain_for_test() == 0


async def test_flush_loop_no_session_factory():
    # If session factory is None, it should bail and log error
    logsvc._queue = asyncio.Queue()
    with patch("dsbx.web.logging.service.log") as mock_log:
        await logsvc._flush_loop(batch_size=10, flush_interval=0.1)
        mock_log.error.assert_called_once_with(
            "dsbx-web: log flush loop started without a DB engine; bailing"
        )


async def test_drain_for_test_success(temp_db):
    await logdb.init_engine(temp_db)
    logsvc._queue = asyncio.Queue()
    entry = make_test_log_entry()
    await logsvc.enqueue_log(entry)

    written = await logsvc.drain_for_test()
    assert written == 1

    # Confirm it was committed to DB
    session_factory = logdb.get_session_factory()
    async with session_factory() as session:
        from sqlalchemy import select

        from dsbx.web.logging.db import RequestLog

        res = await session.execute(select(RequestLog))
        rows = res.scalars().all()
        assert len(rows) == 1
        assert rows[0].backend_name == "test-backend"
        # Check masking was applied
        assert rows[0].request_headers["Authorization"] == "Bea***************123"


async def test_flush_loop_periodical_flushes(temp_db):
    await logdb.init_engine(temp_db)
    # Start service with short interval
    logsvc.start_logging_service(batch_size=2, flush_interval=0.05)
    await asyncio.sleep(0.01)  # let the loop run

    entry1 = make_test_log_entry()
    entry2 = make_test_log_entry()
    entry2.id = "c1234567-89ab-cdef-0123-456789abcde2"

    await logsvc.enqueue_log(entry1)
    await logsvc.enqueue_log(entry2)

    # Wait a bit for the flush loop to run and write them
    await asyncio.sleep(0.15)

    # Verify in DB
    session_factory = logdb.get_session_factory()
    async with session_factory() as session:
        from sqlalchemy import select

        from dsbx.web.logging.db import RequestLog

        res = await session.execute(select(RequestLog))
        rows = res.scalars().all()
        assert len(rows) == 2


async def test_flush_loop_batch_error_keeps_loop_alive(temp_db):
    await logdb.init_engine(temp_db)
    logsvc.start_logging_service(batch_size=2, flush_interval=0.05)
    await asyncio.sleep(0.01)  # let the loop run

    entry1 = make_test_log_entry()

    with (
        patch(
            "dsbx.web.logging.service._write_batch", side_effect=ValueError("DB error")
        ) as mock_write,
        patch("dsbx.web.logging.service.log") as mock_log,
    ):
        await logsvc.enqueue_log(entry1)
        await asyncio.sleep(0.1)
        # Should have attempted write and caught the error
        mock_write.assert_called()
        mock_log.exception.assert_called_with("dsbx-web: log flush batch errored (%d entries)", 1)

        # Loop should still be running
        assert not logsvc._flush_task.done()


async def test_stop_logging_service_flushes_pending(temp_db):
    await logdb.init_engine(temp_db)
    logsvc.start_logging_service(batch_size=10, flush_interval=10.0)  # long interval
    await asyncio.sleep(0.01)  # let the loop run

    entry = make_test_log_entry()
    await logsvc.enqueue_log(entry)

    # Queue should not be empty yet
    assert not logsvc.get_queue().empty()

    # Stopping should force a final flush
    await logsvc.stop_logging_service()

    # Confirm it's stopped and written
    session_factory = logdb.get_session_factory()
    async with session_factory() as session:
        from sqlalchemy import select

        from dsbx.web.logging.db import RequestLog

        res = await session.execute(select(RequestLog))
        rows = res.scalars().all()
        assert len(rows) == 1


async def test_stop_logging_service_shutdown_flush_failure(temp_db):
    await logdb.init_engine(temp_db)
    logsvc.start_logging_service(batch_size=10, flush_interval=10.0)
    await asyncio.sleep(0.01)  # let the loop run

    entry = make_test_log_entry()
    await logsvc.enqueue_log(entry)

    with (
        patch(
            "dsbx.web.logging.service._write_batch", side_effect=Exception("shutdown write error")
        ),
        patch("dsbx.web.logging.service.log") as mock_log,
    ):
        await logsvc.stop_logging_service()
        mock_log.exception.assert_called_once_with("dsbx-web: shutdown log flush failed")


async def test_write_batch_empty():
    # Should exit early without raising or executing database queries
    from dsbx.web.logging.service import _write_batch

    # Pass None as session factory, if it doesn't raise it means it exited early
    await _write_batch(None, [])


async def test_drain_for_test_no_session_factory(temp_db):
    await logdb.init_engine(temp_db)
    logsvc._queue = asyncio.Queue()
    await logsvc.enqueue_log(make_test_log_entry())

    # Mock get_session_factory to return None
    with patch("dsbx.web.logging.service.get_session_factory", return_value=None):
        written = await logsvc.drain_for_test()
        assert written == 0


async def test_drain_for_test_queue_empty_exception(temp_db):
    await logdb.init_engine(temp_db)
    logsvc._queue = asyncio.Queue()
    await logsvc.enqueue_log(make_test_log_entry())

    # Mock Queue.get_nowait to raise QueueEmpty
    original_get = logsvc._queue.get_nowait
    get_nowait_called = False

    def mock_get_nowait():
        nonlocal get_nowait_called
        get_nowait_called = True
        raise asyncio.QueueEmpty()

    logsvc._queue.get_nowait = mock_get_nowait

    try:
        written = await logsvc.drain_for_test()
        assert written == 0
        assert get_nowait_called is True
    finally:
        logsvc._queue.get_nowait = original_get
