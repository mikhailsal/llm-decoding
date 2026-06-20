"""Shared rich rendering helpers for the CLI.

The token renderer's job is to make every byte of a token's text *visible*
in a console column without breaking Rich's markup. Two long-standing
problems made this non-obvious before:

* ``"I"``, ``" I"`` and ``"I "`` are three different tokens, but in a
  column with default padding all three look identical -- whitespace inside
  a cell is indistinguishable from the cell's padding. This module now
  surfaces leading/trailing whitespace with a visible marker (``·`` for
  spaces) and replaces newlines (``↵``), tabs (``→``) and other control
  bytes (``\\xNN``) with explicit escapes. Internal spaces are left intact
  so normal English text still reads naturally.
* Special tokens like ``<|endoftext|>``, ``<|im_start|>``, ``<|fim_prefix|>``
  used to render as plain text and blend into the table. We now colour them
  ``magenta bold`` -- backends that know which ids are special set
  ``TokenCandidate.is_special``; the renderer also pattern-matches
  ``<|...|>`` as a fallback for backends that don't expose the info.
"""

from __future__ import annotations

import math
import re

from decoding_sandbox.core.types import TokenCandidate

SPACE_MARK = "·"  # U+00B7 MIDDLE DOT -- narrow, unobtrusive
NEWLINE_MARK = "↵"
TAB_MARK = "→"
EMPTY_MARK = "<empty>"
SPECIAL_STYLE = "magenta bold"

# Conservative heuristic: tokenizers that follow the "<|name|>" convention
# (Qwen, GPT-style chat tokenizers, FIM, OpenChat, etc.) wrap their reserved
# tokens this way. Matches the whole string -- not a substring -- so a
# legitimate snippet that mentions "<|x|>" inside a longer piece is not
# misclassified.
_SPECIAL_TEXT_RE = re.compile(r"^<\|[A-Za-z0-9_\-\.]+\|>$")


def is_special_text(text: str) -> bool:
    """Heuristic detector for special tokens by their printed text alone."""
    return bool(_SPECIAL_TEXT_RE.match(text))


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


def token_repr(
    text: str,
    width: int | None = None,
    *,
    is_special: bool = False,
) -> str:
    """Render a token so a human can tell ``"I"``, ``" I"``, ``"I "`` apart.

    Pipeline: build the visible string first (no rich markup), truncate it
    if a ``width`` cap is set, *then* wrap in the appropriate style. That
    way truncation never cuts inside a markup tag.

    Rules:

    * Empty text -> ``<empty>`` rendered dim.
    * Special tokens (``is_special=True`` or text matches ``<|...|>``)
      render in magenta bold with no whitespace mangling -- the token
      already is its own escape sequence.
    * Leading and trailing runs of spaces become ``·`` markers; internal
      spaces are left alone so prose reads naturally.
    * ``\\n`` -> ``↵``, ``\\t`` -> ``→``, other C0 control bytes -> ``\\xNN``.
    * Rich's ``[`` is escaped so token text never accidentally opens a tag.
    """
    if text == "":
        # A token that detokenizes to "" is usually a control token (EOS/BOS/
        # PAD) whose printable representation is intentionally empty. If the
        # caller already knows it's special, label it that way; otherwise
        # surface the more neutral <empty>.
        marker = "<special>" if is_special else EMPTY_MARK
        style = SPECIAL_STYLE if is_special else "dim"
        return f"[{style}]{_truncate(marker, width)}[/{style}]"

    if is_special or is_special_text(text):
        safe = text.replace("[", "\\[")
        return f"[{SPECIAL_STYLE}]{_truncate(safe, width)}[/{SPECIAL_STYLE}]"

    n_lead = len(text) - len(text.lstrip(" "))
    n_trail = len(text) - len(text.rstrip(" "))
    core = text.strip(" ")
    # An all-spaces token has lstrip drain it -- in that case every char is
    # both "leading" and "trailing"; count it once.
    if not core:
        n_trail = 0
    core = core.replace("[", "\\[")
    core = core.replace("\n", NEWLINE_MARK).replace("\t", TAB_MARK)
    core = _escape_other_controls(core)
    shown = (SPACE_MARK * n_lead) + core + (SPACE_MARK * n_trail)
    return _truncate(shown, width)


def _escape_other_controls(s: str) -> str:
    out: list[str] = []
    for ch in s:
        cp = ord(ch)
        if cp < 0x20 and ch not in ("\n", "\t"):
            # NB: \n and \t already turned into NEWLINE_MARK / TAB_MARK above.
            out.append(f"\\x{cp:02x}")
        else:
            out.append(ch)
    return "".join(out)


def _truncate(s: str, width: int | None) -> str:
    if width is None or len(s) <= width:
        return s
    return s[: width - 1] + "…"


def candidate_brief(c: TokenCandidate) -> str:
    return f"{token_repr(c.text, 12, is_special=c.is_special)!s} {fmt_prob(c.prob)}"


def watch_cell(c: TokenCandidate | None) -> str:
    if c is None or math.isnan(c.prob):
        return "[dim]<top-k[/dim]"
    rank = f"#{c.rank}" if c.rank >= 0 else "?"
    return f"{fmt_prob(c.prob)} [dim]{rank}[/dim]"
