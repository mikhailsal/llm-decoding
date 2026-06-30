"""Helpers shared across CLI subcommands.

These functions historically lived in ``dsbx.cli.app``; they were
extracted so each command module stays small. They print through
``app.console`` so the test harness's console monkeypatch still applies.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from rich.console import Console

from dsbx.cli import app
from dsbx.cli.timing import Timing
from dsbx.core.backend import Backend
from dsbx.core.config import Config


@dataclass(frozen=True)
class WatchTarget:
    """One column in ``inspect``'s "watch" view: a labeled token id.

    Three legal sources, distinguished by the ``label`` prefix:

    * ``"text:<repr>"`` -- the user passed ``--watch TEXT`` and we resolved
      the first token id of ``TEXT``. ``repr`` is included so the column
      header shows the user-visible string with quotes.
    * ``"id=<N>[ <piece>]"`` -- the user passed ``--watch-id N``. The
      piece text is appended if non-empty, so the header reads
      ``id=42 ' the'`` instead of just ``id=42``.
    * ``"EOS:<N>"`` -- the user passed ``--watch-eos`` and we expanded it
      to every id in ``backend.capabilities.eos_token_ids``.

    The renderer uses the label literally in the column header, so the
    distinction is preserved end-to-end without any other branching.
    """

    label: str
    token_id: int


def _resolve_watch(backend, watch: list[str]) -> list[WatchTarget]:
    """Resolve each ``--watch TEXT`` string to a single token id.

    Warns and skips empty or multi-token tokenizations -- a multi-token
    watch is impossible to track at the per-position level (which id would
    we read?). The user's exact input is preserved in the label so the
    column header is recognizable.
    """
    resolved: list[WatchTarget] = []
    for w in watch:
        ids = backend.tokenize(w)
        if not ids:
            app.console.print(f"[yellow]watch {w!r}: tokenizes to nothing, skipped[/yellow]")
            continue
        if len(ids) > 1:
            app.console.print(
                f"[yellow]watch {w!r}: {len(ids)} tokens; watching first "
                f"({backend.piece(ids[0])!r}). Try a leading space.[/yellow]"
            )
        resolved.append(WatchTarget(label=f"text:{w!r}", token_id=int(ids[0])))
    return resolved


def _resolve_watch_ids(backend, ids: list[int]) -> list[WatchTarget]:
    """Wrap each raw id as a WatchTarget with a descriptive label.

    The piece text (when non-empty) is appended so users can see at a
    glance which token they pinned -- helpful for sanity-checking, e.g.
    ``--watch-id 1234`` reading ``id=1234 ' Paris'`` confirms the right
    one.
    """
    from dsbx.cli import render as _render

    out: list[WatchTarget] = []
    for raw in ids:
        try:
            tid = int(raw)
        except (TypeError, ValueError):
            app.console.print(f"[yellow]watch-id {raw!r}: not an integer, skipped[/yellow]")
            continue
        piece = backend.piece(tid) if hasattr(backend, "piece") else ""
        suffix = f" {_render.token_repr(piece, 12, is_special=True)}" if piece else ""
        out.append(WatchTarget(label=f"id={tid}{suffix}", token_id=tid))
    return out


def _resolve_watch_eos(backend) -> list[WatchTarget]:
    """Expand ``--watch-eos`` to one WatchTarget per advertised EOS id.

    Backends that don't expose EOS (HTTP llama.cpp, cloud providers) yield
    a friendly warning and an empty result -- the user asked for something
    the backend can't give them, and silent "nothing happens" would be a
    debugging pitfall.
    """
    eos_ids = list(backend.capabilities.eos_token_ids)
    if not eos_ids:
        app.console.print(
            "[yellow]--watch-eos: this backend does not expose EOS ids "
            "(Capabilities.eos_token_ids is empty); skipped.[/yellow]"
        )
        return []
    return [WatchTarget(label=f"EOS:{tid}", token_id=int(tid)) for tid in eos_ids]


def _collect_watch_targets(
    backend,
    *,
    texts: list[str],
    ids: list[int],
    eos: bool,
) -> list[WatchTarget]:
    """Merge text/id/eos watches into one ordered, deduped list.

    Order is preserved (texts first, then ids, then EOS expansions) so
    column ordering in the table matches the user's flag order on the CLI.
    Dedup is by ``token_id``: if the same id arrives via two different
    flags (e.g. ``--watch ' Paris' --watch-id 1234`` and they happen to
    collide), the first wins, keeping its label.
    """
    merged: list[WatchTarget] = []
    seen: set[int] = set()
    sources = [
        _resolve_watch(backend, texts),
        _resolve_watch_ids(backend, ids),
        _resolve_watch_eos(backend) if eos else [],
    ]
    for batch in sources:
        for target in batch:
            if target.token_id in seen:
                continue
            seen.add(target.token_id)
            merged.append(target)
    return merged


def _resolve_stop_ids(backend, stop: list[str]) -> list[tuple[str, int]]:
    """Map each stop string to a single token id (skip + warn if multi-token).

    Generation halts the moment any chosen token matches one of these ids. A
    multi-token stop string is impossible to detect at the per-token level, so
    we warn and ignore it -- the user should prefer a single-token stop (e.g.
    a newline or a specific punctuation token).
    """
    resolved: list[tuple[str, int]] = []
    for s in stop:
        ids = backend.tokenize(s)
        if not ids:
            app.console.print(f"[yellow]stop {s!r}: tokenizes to nothing, skipped[/yellow]")
            continue
        if len(ids) > 1:
            app.console.print(
                f"[yellow]stop {s!r}: {len(ids)} tokens; cannot match per-step, "
                f"skipped. Try a single-token stop like '\\n'.[/yellow]"
            )
            continue
        resolved.append((s, ids[0]))
    return resolved


def _build_backend_with_load_timing(
    name: str, cfg: Config, *, model: str | None, timing: Timing | None
) -> Backend:
    """Build a backend and (optionally) record its load time as a phase."""
    from dsbx.core.factory import build_backend

    app.console.print(f"[dim]building backend '{name}'...[/dim]")
    if timing is None:
        return build_backend(name, cfg, model=model)
    with timing.phase("backend load"):
        return build_backend(name, cfg, model=model)


def _print_backend_banner(backend: Backend, *, out: Console | None = None) -> None:
    """Print the capability banner that ``inspect`` historically led with.

    Also surfaces the same backend-specific notes (no native whole-context,
    top-k only, llamacpp HTTP nudge to llamacpp-py). ``out`` defaults to the
    module-level rich app.console; the session REPL passes its own buffered
    app.console so meta commands write where the caller expects.
    """
    out = out or app.console
    caps = backend.capabilities
    out.print(
        f"backend: [cyan]{caps.name}[/cyan]  "
        f"full_vocab={caps.full_vocab}  prompt_logprobs={caps.prompt_logprobs}  "
        f"max_top_logprobs={caps.max_top_logprobs}"
    )
    if caps.eos_token_ids:
        # Help the user understand "how is EOS transmitted?" -- list the
        # token ids the backend believes terminate generation, along with
        # the pieces those ids decode to. Pieces are rendered with the
        # full token-repr rules so special markers stay visible.
        from dsbx.cli import render as _render

        pieces: list[str] = []
        for tid in caps.eos_token_ids:
            text = backend.piece(tid) if hasattr(backend, "piece") else ""
            pieces.append(f"{tid}={_render.token_repr(text, 16, is_special=True)}")
        out.print(f"[dim]EOS ids: {', '.join(pieces)}[/dim]")
    else:
        out.print("[dim]EOS ids: <not exposed by this backend>[/dim]")
    if caps.notes:
        out.print(f"[dim]{caps.notes}[/dim]")
    if not caps.full_vocab:
        out.print(
            "[dim]note: top-k backend -- a token's probability is shown only if it "
            "is within the returned top-k (others read '<top-k').[/dim]"
        )
    if not caps.full_vocab and not caps.prompt_logprobs:
        if backend.__class__.__name__ == "LlamaCppBackend":
            out.print(
                "[dim]note: this backend exposes top-k only and derives "
                "whole-context one position at a time (cheap with cache_prompt). "
                "For FULL vocab on the same GGUF, use --backend llamacpp-py.[/dim]"
            )
        else:
            out.print(
                "[yellow]note: this backend has no native whole-context "
                "logprobs; each prompt position is re-evaluated separately, "
                "which is genuinely slow for chat-only cloud providers.[/yellow]"
            )


@contextmanager
def _null_phase():
    """No-op context manager used when timing is disabled."""
    yield


def _maybe_phase(timing: Timing | None, name: str, *, tokens: int | None = None):
    """Return ``timing.phase(...)`` or a no-op context manager when None."""
    if timing is not None:
        return timing.phase(name, tokens=tokens)
    return _null_phase()


def _print_candidates(steps, max_positions: int, top_k: int) -> None:
    from dsbx.cli import render

    app.console.print(f"\n[bold]Top-{top_k} candidates (first {max_positions} positions)[/bold]")
    for st in steps[:max_positions]:
        ctx = render.token_repr(st.context_text or "", 14)
        line = f"[cyan]pos {st.position}[/cyan] after {ctx!r}: "
        line += "  ".join(
            f"{render.token_repr(c.text, 10, is_special=c.is_special)!s}={render.fmt_prob(c.prob)}"
            for c in st.candidates
        )
        app.console.print(line)
