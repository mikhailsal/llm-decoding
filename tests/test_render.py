"""Tests for the rich rendering helpers (confidence colors, formatting)."""

from __future__ import annotations

import math

from decoding_sandbox.cli import render
from decoding_sandbox.core.types import TokenCandidate


def test_confidence_style_thresholds() -> None:
    assert render.confidence_style(0.95) == "bold green"
    assert render.confidence_style(0.70) == "bold green"  # inclusive boundary
    assert render.confidence_style(0.69) == "green"
    assert render.confidence_style(0.40) == "green"
    assert render.confidence_style(0.39) == "yellow"
    assert render.confidence_style(0.20) == "yellow"
    assert render.confidence_style(0.19) == "red"
    assert render.confidence_style(0.05) == "red"
    assert render.confidence_style(0.04) == "bright_red"
    assert render.confidence_style(0.0) == "bright_red"
    assert render.confidence_style(float("nan")) == "dim"


def test_fmt_prob_renders_percentage_with_style() -> None:
    out = render.fmt_prob(0.42)

    assert "42." in out
    assert "%" in out
    assert "green" in out  # the matching style is embedded


def test_fmt_prob_nan_is_dim_question_mark() -> None:
    out = render.fmt_prob(float("nan"))
    assert "dim" in out
    assert "?" in out


def test_token_repr_escapes_whitespace_and_brackets() -> None:
    out = render.token_repr("a\nb\t[c]")
    assert "\\n" in out
    assert "\\t" in out
    assert "\\[" in out


def test_token_repr_truncates_at_width() -> None:
    out = render.token_repr("abcdefghij", width=5)
    assert len(out) == 5
    assert out.endswith("…")


def test_token_repr_no_truncation_when_width_none() -> None:
    out = render.token_repr("hello", width=None)
    assert out == "hello"


def test_candidate_brief_combines_text_and_prob() -> None:
    c = TokenCandidate(1, "x", math.log(0.5), 0)
    out = render.candidate_brief(c)
    assert "x" in out
    assert "%" in out


def test_watch_cell_none_or_nan_shows_top_k_marker() -> None:
    assert "<top-k" in render.watch_cell(None)

    nan_cand = TokenCandidate(1, "x", float("nan"), -1)
    assert "<top-k" in render.watch_cell(nan_cand)


def test_watch_cell_known_token_shows_prob_and_rank() -> None:
    c = TokenCandidate(1, "x", math.log(0.4), 3)
    out = render.watch_cell(c)
    assert "#3" in out
    assert "%" in out
