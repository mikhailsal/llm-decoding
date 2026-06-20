"""Shared rich rendering helpers for the CLI (confidence colors, token display)."""

from __future__ import annotations

import math

from decoding_sandbox.core.types import TokenCandidate


def confidence_style(prob: float) -> str:
    """Map a probability to a rich color (the 'confidence' encoding)."""
    if math.isnan(prob):
        return "dim"
    if prob >= 0.70:
        return "bold green"
    if prob >= 0.40:
        return "green"
    if prob >= 0.20:
        return "yellow"
    if prob >= 0.05:
        return "red"
    return "bright_red"


def fmt_prob(prob: float) -> str:
    """Colored percentage, e.g. '[green]42.3%[/green]'. NaN -> unknown."""
    if math.isnan(prob):
        return "[dim]   ?  [/dim]"
    return f"[{confidence_style(prob)}]{prob:6.2%}[/{confidence_style(prob)}]"


def token_repr(text: str, width: int | None = None) -> str:
    """Make whitespace visible and escape rich markup."""
    shown = text.replace("\n", "\\n").replace("\t", "\\t")
    shown = shown.replace("[", "\\[")
    if width is not None and len(shown) > width:
        shown = shown[: width - 1] + "…"
    return shown


def candidate_brief(c: TokenCandidate) -> str:
    return f"{token_repr(c.text, 12)!s} {fmt_prob(c.prob)}"


def watch_cell(c: TokenCandidate | None) -> str:
    if c is None or math.isnan(c.prob):
        return "[dim]<top-k[/dim]"
    rank = f"#{c.rank}" if c.rank >= 0 else "?"
    return f"{fmt_prob(c.prob)} [dim]{rank}[/dim]"
