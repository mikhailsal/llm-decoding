"""Unit tests for the usage accumulator helpers."""

from __future__ import annotations

from dsbx.core.usage import (
    UsageSink,
    add_note,
    make_sink,
    record_request,
    record_tokens,
)


def test_make_sink_initializes_all_fields() -> None:
    sink = make_sink()
    assert sink["requests"] == 0
    assert sink["prompt_tokens"] is None
    assert sink["completion_tokens"] is None
    assert sink["total_tokens"] is None
    assert sink["notes"] == []


def test_record_request_none_is_noop_and_increments() -> None:
    sink: UsageSink = {}
    record_request(None)
    record_request(sink)
    record_request(sink, n=2)
    assert sink["requests"] == 3


def test_record_tokens_none_sink_is_noop() -> None:
    # Should not raise; just a guard
    record_tokens(None, prompt_tokens=3)


def test_record_tokens_increments_fields() -> None:
    sink: UsageSink = {}
    record_tokens(sink, prompt_tokens=2)
    record_tokens(sink, completion_tokens=5)
    record_tokens(sink, total_tokens=1)
    assert sink["prompt_tokens"] == 2
    assert sink["completion_tokens"] == 5
    assert sink["total_tokens"] == 1

    # Second call accumulates
    record_tokens(sink, prompt_tokens=1, completion_tokens=1, total_tokens=1)
    assert sink["prompt_tokens"] == 3
    assert sink["completion_tokens"] == 6
    assert sink["total_tokens"] == 2


def test_add_note_none_or_empty_is_noop() -> None:
    sink: UsageSink = {}
    add_note(None, "x")
    add_note(sink, "")
    assert "notes" not in sink


def test_add_note_appends_and_dedupes() -> None:
    sink: UsageSink = {}
    add_note(sink, "first")
    add_note(sink, "second")
    add_note(sink, "first")  # duplicate ignored
    assert sink["notes"] == ["first", "second"]


def test_record_perf_metrics_merges_and_overwrites() -> None:
    from dsbx.core.usage import record_perf_metrics

    sink: UsageSink = {}
    record_perf_metrics(sink, {"a": 1})
    record_perf_metrics(sink, {"b": 2, "a": 99})  # last write wins on overlap
    assert sink["perf_metrics"] == {"a": 99, "b": 2}

    # None or non-dict input is a no-op (cover the guard lines)
    record_perf_metrics(None, {"x": 1})
    record_perf_metrics(sink, None)
    record_perf_metrics(sink, "not-a-dict")
    record_perf_metrics(sink, 123)
    assert sink["perf_metrics"] == {"a": 99, "b": 2}


def test_record_raw_output_last_write_wins() -> None:
    from dsbx.core.usage import record_raw_output

    sink: UsageSink = {}
    record_raw_output(sink, {"k": "v1"})
    record_raw_output(sink, {"k": "v2"})
    assert sink["raw_output"] == {"k": "v2"}

    record_raw_output(None, {"x": 1})  # no crash
    record_raw_output(sink, None)
    record_raw_output(sink, "bad")
    record_raw_output(sink, ["list"])
    assert sink["raw_output"] == {"k": "v2"}
