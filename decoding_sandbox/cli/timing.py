"""Tiny timing helpers for CLI commands.

A ``Timing`` tracker collects named phases via a context manager, then renders
a one-line summary plus per-phase tokens-per-second for any phase that was
asked to count tokens. Cheap and stdlib-only: ``time.perf_counter`` + a list.

We expose it as a small object instead of decorators because most commands
have a couple of phases that mean different things ("prompt eval" vs "decode"
vs "verify pass" for speculative). The caller decides which token count is
the right divisor for each phase -- e.g. for ``inspect`` the divisor is the
number of prompt tokens, for ``generate`` decode it's the number of new
tokens, and for ``spec`` it's the emitted tokens.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Phase:
    name: str
    seconds: float
    tokens: int | None = None  # divisor for the tok/s column; None = omit

    @property
    def tps(self) -> float | None:
        if self.tokens is None or self.seconds <= 0:
            return None
        return self.tokens / self.seconds


@dataclass
class Timing:
    """Tiny tracker for phase durations + tokens-per-second.

    Usage:

        timing = Timing()
        with timing.phase("prompt eval", tokens=n_prompt_tokens):
            steps = backend.score_prompt(prompt, top_k)
        with timing.phase("decode", tokens=n_gen_tokens):
            ...
        console.print(timing.render())
    """

    phases: list[Phase] = field(default_factory=list)

    @contextmanager
    def phase(self, name: str, *, tokens: int | None = None) -> Iterator[None]:
        t0 = time.perf_counter()
        # Reserve a slot so even if the body raises we still record what
        # happened (handy for diagnosing crashes mid-decode).
        p = Phase(name=name, seconds=0.0, tokens=tokens)
        self.phases.append(p)
        try:
            yield
        finally:
            p.seconds = time.perf_counter() - t0

    @property
    def total_seconds(self) -> float:
        return sum(p.seconds for p in self.phases)

    def set_tokens(self, name: str, tokens: int) -> None:
        """Patch a phase's token count after the fact (useful when the count
        is only known after the body has run, e.g. for generate())."""
        for p in self.phases:
            if p.name == name:
                p.tokens = tokens
                return

    def record(self, name: str, seconds: float, *, tokens: int | None = None) -> None:
        """Append an already-measured phase.

        Useful when the natural API isn't a context manager around a block --
        e.g. timing the prompt-eval part of ``generate`` which is implicit in
        the first iteration of the generator.
        """
        self.phases.append(Phase(name=name, seconds=seconds, tokens=tokens))

    def render(self, *, prefix: str = "timing") -> str:
        """One-line summary suitable for a rich.Console.print call.

        Example:
            timing: prompt eval 7 tok in 487 ms = 14.4 tok/s | total 510 ms
        """
        if not self.phases:
            return f"[dim]{prefix}: (no phases recorded)[/dim]"
        parts: list[str] = []
        for p in self.phases:
            if p.tokens is None:
                parts.append(f"{p.name} {_fmt_seconds(p.seconds)}")
                continue
            tps = p.tps
            tps_str = f" = {tps:.1f} tok/s" if tps is not None else ""
            parts.append(f"{p.name} {p.tokens} tok in {_fmt_seconds(p.seconds)}{tps_str}")
        parts.append(f"total {_fmt_seconds(self.total_seconds)}")
        return "[dim]" + prefix + ": " + " | ".join(parts) + "[/dim]"


def _fmt_seconds(s: float) -> str:
    """Render seconds with a sensible unit (ms below 1 s, s above)."""
    if s < 1.0:
        return f"{s * 1000:.0f} ms"
    if s < 10.0:
        return f"{s:.2f} s"
    return f"{s:.1f} s"


__all__ = ["Phase", "Timing"]
