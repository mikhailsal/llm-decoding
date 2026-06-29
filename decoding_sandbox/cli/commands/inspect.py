"""``dsbx inspect`` -- per-token confidence + watch-token highlighting."""

from __future__ import annotations

import argparse

from rich.table import Table

from decoding_sandbox.cli import app
from decoding_sandbox.cli._shared import (
    _build_backend_with_load_timing,
    _collect_watch_targets,
    _maybe_phase,
    _print_backend_banner,
    _print_candidates,
)
from decoding_sandbox.cli.timing import Timing
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config


def cmd_inspect(
    args: argparse.Namespace,
    cfg: Config,
    *,
    backend: Backend | None = None,
    show_banner: bool = True,
) -> int:
    """Per-token confidence inspection of ``args.prompt``.

    If ``backend`` is provided (e.g. by the long-lived ``session`` REPL), it
    is reused and not closed; otherwise this function owns the backend it
    builds and closes it at the end.
    """
    from decoding_sandbox.cli import render

    own_backend = backend is None
    if own_backend:
        rc = app._run_preflight(cfg, skip=getattr(args, "skip_preflight", False))
        if rc is not None:
            return rc

    timing = None if getattr(args, "no_timing", False) else Timing()
    try:
        if own_backend:
            name = args.backend or cfg.default_backend
            backend = _build_backend_with_load_timing(name, cfg, model=args.model, timing=timing)
        if show_banner:
            _print_backend_banner(backend)

        watch = _collect_watch_targets(
            backend,
            texts=list(args.watch or []),
            ids=list(getattr(args, "watch_id", []) or []),
            eos=bool(getattr(args, "watch_eos", False)),
        )
        caps = backend.capabilities
        prompt_tokens = backend.tokenize(args.prompt)
        generated_only_inspect = (
            backend.__class__.__name__ == "OpenAICompatBackend" and not caps.prompt_logprobs
        )
        if generated_only_inspect:
            app.console.print(
                "[yellow]note: this backend cannot score prompt tokens; showing the "
                "next-token distribution after the prompt instead.[/yellow]"
            )
            tok_count = 1
            with _maybe_phase(timing, "next-token distribution", tokens=tok_count):
                step = backend.next_distribution(prompt_tokens, top_k=args.top_k)
            step.context_text = args.prompt
            step.watched = {t.token_id: backend.lookup_watch(step, t.token_id) for t in watch}
            steps = [step]
            title = f"Next-token inspection: {args.prompt!r}"
        else:
            tok_count = len(prompt_tokens)
            with _maybe_phase(timing, "score_prompt", tokens=tok_count):
                steps = backend.score_prompt(
                    args.prompt,
                    top_k=args.top_k,
                    watch_ids=[t.token_id for t in watch],
                )
            title = f"Context inspection: {args.prompt!r}"

        table = Table(title=title)
        table.add_column("pos", justify="right")
        table.add_column("context -> next")
        table.add_column("p(next)", justify="right")
        table.add_column("rank", justify="right")
        table.add_column("confidence (top-1)")
        for target in watch:
            table.add_column(f"watch {target.label}")

        for st in steps:
            ctx = render.token_repr(st.context_text or "", 14)
            is_trailing = st.chosen is None
            if not is_trailing:
                nxt = render.token_repr(st.chosen.text, 14, is_special=st.chosen.is_special)
                p_next = render.fmt_prob(st.chosen.prob)
                rank = f"#{st.chosen.rank}" if st.chosen.rank >= 0 else "[dim]?[/dim]"
                pos_cell = str(st.position)
            else:
                # The trailing "predict next" row: there is no actual next
                # token to score against, but the watched columns and the
                # top-1 confidence are real predictions worth seeing. Mark
                # the row visibly so it isn't mistaken for a scored step.
                nxt = "[dim]?[/dim]"
                p_next = "[dim]?[/dim]"
                rank = "[dim]?[/dim]"
                pos_cell = f"{st.position} [dim](next)[/dim]"
            top = st.top
            conf = (
                f"{render.fmt_prob(st.confidence)} "
                f"{render.token_repr(top.text, 12, is_special=top.is_special)!s}"
                if top
                else "-"
            )
            row = [pos_cell, f"{ctx} -> {nxt}", p_next, rank, conf]
            for target in watch:
                row.append(render.watch_cell(st.watched.get(target.token_id)))
            table.add_row(*row)

        app.console.print(table)

        if args.candidates:
            _print_candidates(steps, args.candidates, args.top_k)

        if timing is not None:
            app.console.print(timing.render())
        return 0
    finally:
        if own_backend and backend is not None:
            backend.close()
