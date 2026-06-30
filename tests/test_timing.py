"""Tests for the CLI timing tracker.

We don't need a real clock for any of this -- we inject elapsed times by
``record(...)``, and only smoke-test the context-manager path with a small
synthetic sleep so the suite stays fast.
"""

from __future__ import annotations

import time

from dsbx.cli.timing import Phase, Timing, _fmt_seconds


# --------------------------------------------------------------------------- #
# Phase math
# --------------------------------------------------------------------------- #
def test_phase_tps_is_none_when_tokens_is_none() -> None:
    p = Phase("x", 1.0, tokens=None)
    assert p.tps is None


def test_phase_tps_divides_tokens_by_seconds() -> None:
    p = Phase("x", 0.5, tokens=10)
    assert p.tps == 20.0


def test_phase_tps_guards_zero_seconds() -> None:
    p = Phase("x", 0.0, tokens=10)
    assert p.tps is None


# --------------------------------------------------------------------------- #
# Recording phases
# --------------------------------------------------------------------------- #
def test_record_appends_a_phase_with_optional_tokens() -> None:
    t = Timing()
    t.record("a", 0.1)
    t.record("b", 0.2, tokens=4)

    assert [p.name for p in t.phases] == ["a", "b"]
    assert t.phases[0].tokens is None
    assert t.phases[1].tps == 20.0


def test_set_tokens_patches_an_existing_phase() -> None:
    t = Timing()
    t.record("decode", 1.0)
    t.set_tokens("decode", 50)
    assert t.phases[0].tps == 50.0


def test_set_tokens_is_a_noop_when_name_doesnt_match() -> None:
    t = Timing()
    t.record("decode", 1.0)
    t.set_tokens("other", 99)
    assert t.phases[0].tokens is None


def test_total_seconds_sums_phase_durations() -> None:
    t = Timing()
    t.record("a", 1.5)
    t.record("b", 0.25)
    assert t.total_seconds == 1.75


# --------------------------------------------------------------------------- #
# Context-manager phase
# --------------------------------------------------------------------------- #
def test_phase_context_manager_records_elapsed_time() -> None:
    t = Timing()
    with t.phase("work", tokens=2):
        time.sleep(0.01)
    assert len(t.phases) == 1
    p = t.phases[0]
    assert p.name == "work"
    assert p.tokens == 2
    assert p.seconds >= 0.01
    assert p.tps is not None and p.tps > 0


def test_phase_context_manager_records_even_on_exception() -> None:
    t = Timing()
    try:
        with t.phase("explodes"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert len(t.phases) == 1
    assert t.phases[0].seconds >= 0


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_render_empty_timing_says_so() -> None:
    out = Timing().render()
    assert "no phases recorded" in out


def test_render_includes_phase_names_and_tokens_per_second() -> None:
    t = Timing()
    t.record("prompt eval", 0.5, tokens=10)  # = 20 tok/s
    t.record("decode", 1.0, tokens=50)  # = 50 tok/s

    out = t.render()
    assert "prompt eval 10 tok in" in out
    assert "20.0 tok/s" in out
    assert "decode 50 tok in" in out
    assert "50.0 tok/s" in out
    assert "total " in out


def test_render_phase_without_tokens_omits_tok_s() -> None:
    t = Timing()
    t.record("backend load", 12.0)
    out = t.render()

    assert "backend load" in out
    assert "tok/s" not in out


def test_render_prefix_is_customizable() -> None:
    t = Timing()
    t.record("startup", 1.0)
    assert "startup-bench" in t.render(prefix="startup-bench")


# --------------------------------------------------------------------------- #
# Seconds formatting
# --------------------------------------------------------------------------- #
def test_fmt_seconds_uses_ms_under_one_second() -> None:
    assert _fmt_seconds(0.234) == "234 ms"


def test_fmt_seconds_uses_two_decimals_in_low_seconds_range() -> None:
    assert _fmt_seconds(2.5) == "2.50 s"


def test_fmt_seconds_uses_one_decimal_in_high_seconds_range() -> None:
    assert _fmt_seconds(45.0) == "45.0 s"
