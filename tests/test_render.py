"""Tests for the rich rendering helpers (confidence colors, formatting)."""

from __future__ import annotations

import math

from decoding_sandbox.cli import render
from decoding_sandbox.cli.render import (
    EMPTY_MARK,
    NEWLINE_MARK,
    SPACE_MARK,
    SPECIAL_STYLE,
    TAB_MARK,
    is_special_text,
)
from decoding_sandbox.core.types import TokenCandidate


# --------------------------------------------------------------------------- #
# Confidence colours / prob formatting
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# token_repr -- the core "make invisible visible" function
# --------------------------------------------------------------------------- #
def test_token_repr_keeps_normal_text_intact() -> None:
    assert render.token_repr("hello") == "hello"


def test_token_repr_distinguishes_I_from_space_I() -> None:
    """The motivating case: ``"I"`` vs ``" I"`` vs ``"I "`` must not collapse."""
    bare = render.token_repr("I")
    lead = render.token_repr(" I")
    trail = render.token_repr("I ")

    assert bare == "I"
    assert lead == SPACE_MARK + "I"
    assert trail == "I" + SPACE_MARK
    assert bare != lead != trail


def test_token_repr_marks_multiple_leading_spaces() -> None:
    out = render.token_repr("   abc")
    assert out == (SPACE_MARK * 3) + "abc"


def test_token_repr_marks_multiple_trailing_spaces() -> None:
    out = render.token_repr("abc   ")
    assert out == "abc" + (SPACE_MARK * 3)


def test_token_repr_leaves_internal_spaces_intact() -> None:
    """Don't mark internal spaces -- that would make prose unreadable."""
    out = render.token_repr("a b")
    assert out == "a b"


def test_token_repr_renders_only_spaces_as_dots() -> None:
    out = render.token_repr("    ")
    assert out == SPACE_MARK * 4


def test_token_repr_renders_empty_with_dim_marker() -> None:
    out = render.token_repr("")
    assert EMPTY_MARK in out
    assert "dim" in out


def test_token_repr_empty_special_uses_special_label() -> None:
    """An EOS/BOS that detokenizes to "" is a control token, not an accident.
    When the caller flags it special we say so instead of the neutral
    ``<empty>`` -- the same byte stream means different things in those
    two cases."""
    out = render.token_repr("", is_special=True)
    assert "<special>" in out
    assert SPECIAL_STYLE in out


def test_token_repr_replaces_newline_and_tab() -> None:
    out = render.token_repr("a\nb\tc")
    assert NEWLINE_MARK in out
    assert TAB_MARK in out
    assert "a" in out and "b" in out and "c" in out


def test_token_repr_escapes_other_control_characters() -> None:
    out = render.token_repr("a\x07b\x1fc")
    assert "\\x07" in out
    assert "\\x1f" in out


def test_token_repr_escapes_left_bracket_for_rich_markup() -> None:
    out = render.token_repr("[c]")
    assert "\\[" in out  # rich requires escaped opening bracket


def test_token_repr_truncates_long_text_with_ellipsis() -> None:
    out = render.token_repr("abcdefghij", width=5)
    assert out.endswith("…")
    assert len(out) == 5


def test_token_repr_no_truncation_when_below_width() -> None:
    assert render.token_repr("hello", width=10) == "hello"


def test_token_repr_special_token_uses_magenta_style() -> None:
    """is_special=True wraps in magenta bold regardless of whitespace content."""
    out = render.token_repr("<|endoftext|>", is_special=True)
    assert SPECIAL_STYLE in out
    assert "<|endoftext|>" in out


def test_token_repr_auto_detects_special_text_pattern() -> None:
    """``<|name|>`` matches even without is_special=True."""
    out = render.token_repr("<|im_end|>")
    assert SPECIAL_STYLE in out


def test_token_repr_does_not_falsely_match_partial_pattern() -> None:
    """Token text that merely *contains* ``<|...|>`` is NOT special."""
    out = render.token_repr("said <|im_end|> then")
    assert SPECIAL_STYLE not in out


def test_is_special_text_recognizes_common_patterns() -> None:
    assert is_special_text("<|endoftext|>")
    assert is_special_text("<|im_start|>")
    assert is_special_text("<|fim_prefix|>")
    assert not is_special_text("foo")
    assert not is_special_text("<|endoftext|> bar")
    assert not is_special_text("")


# --------------------------------------------------------------------------- #
# Candidate / watch cells now thread is_special through
# --------------------------------------------------------------------------- #
def test_candidate_brief_combines_text_and_prob() -> None:
    c = TokenCandidate(1, "x", math.log(0.5), 0)
    out = render.candidate_brief(c)
    assert "x" in out
    assert "%" in out


def test_candidate_brief_marks_special_tokens() -> None:
    c = TokenCandidate(1, "<|endoftext|>", math.log(0.5), 0, is_special=True)
    out = render.candidate_brief(c)
    assert SPECIAL_STYLE in out


def test_watch_cell_none_or_nan_shows_top_k_marker() -> None:
    assert "<top-k" in render.watch_cell(None)

    nan_cand = TokenCandidate(1, "x", float("nan"), -1)
    assert "<top-k" in render.watch_cell(nan_cand)


def test_watch_cell_known_token_shows_prob_and_rank() -> None:
    c = TokenCandidate(1, "x", math.log(0.4), 3)
    out = render.watch_cell(c)
    assert "#3" in out
    assert "%" in out
