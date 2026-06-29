"""``dsbx generate`` -- decode with a chosen/custom sampler, diff vs greedy."""

from __future__ import annotations

import argparse

from rich.table import Table

from decoding_sandbox.cli import app
from decoding_sandbox.cli._shared import (
    _build_backend_with_load_timing,
    _resolve_stop_ids,
)
from decoding_sandbox.cli.timing import Timing
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config


def cmd_generate(
    args: argparse.Namespace,
    cfg: Config,
    *,
    backend: Backend | None = None,
    show_banner: bool = True,
) -> int:
    import random

    from decoding_sandbox.cli import render
    from decoding_sandbox.core import samplers
    from decoding_sandbox.core.engine import generate

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
            app.console.print(f"backend: [cyan]{backend.capabilities.name}[/cyan]")

        # Sampler construction (built-in or custom plug-in). ``params`` is
        # only populated for the built-in path -- we keep it around even
        # for ``custom`` so the dispatch logic below has one place to look.
        params: dict = {}
        if args.sampler == "custom":
            if not args.custom_file:
                app.console.print(
                    "[red]--sampler custom requires --custom-file path.py[:func][/red]"
                )
                return 2
            sampler = samplers.load_custom(args.custom_file)
            sampler_name = f"custom({args.custom_file})"
        else:
            params = dict(
                temperature=args.temperature,
                top_k=args.sampler_top_k,
                top_p=args.top_p,
                min_p=args.min_p,
                typical_p=args.typical_p,
            )
            params = {k: v for k, v in params.items() if v is not None}
            sampler = samplers.make_sampler(args.sampler, **params)
            sampler_name = args.sampler

        rng = random.Random(args.seed)
        stop_ids = _resolve_stop_ids(backend, args.stop or [])

        # Stream from the server when the backend supports it AND the
        # sampler is a built-in (custom samplers can't run server-side
        # because the server has no way to ingest arbitrary client code).
        # Falling back to the per-step loop keeps custom samplers fully
        # functional, just slower over the network.
        use_remote_stream = hasattr(backend, "stream_generate") and args.sampler != "custom"
        transport = "remote-stream" if use_remote_stream else "in-process"
        app.console.print(
            f"sampler: [magenta]{sampler_name}[/magenta]  seed={args.seed}  "
            f"max_tokens={args.max_tokens}  [dim]({transport})[/dim]"
            + (f"  stop={[s for s, _ in stop_ids]}" if stop_ids else "")
            + "\n"
        )

        table = Table(title=f"generate {args.prompt!r}")
        table.add_column("step", justify="right")
        table.add_column("chosen")
        table.add_column("p(chosen)", justify="right")
        table.add_column("vs greedy")
        table.add_column("kept", justify="right")
        table.add_column("sampler")
        table.add_column("top candidates")

        # Two phases users care about:
        #   1) prompt eval + first token  -- latency until streaming starts
        #   2) subsequent decode          -- per-new-token cost
        # The split happens at the first ``gs`` yielded.
        import time as _time

        chosen_ids: list[int] = []
        loop_start = _time.perf_counter()
        first_token_at: float | None = None
        last_stop_reason: str | None = None
        last_chosen_text: str = ""

        if use_remote_stream:
            step_iter = backend.stream_generate(  # type: ignore[attr-defined]
                args.prompt,
                sampler_name=args.sampler,
                sampler_params=params,
                max_tokens=args.max_tokens,
                top_k=args.top_k,
                stop_ids=[tid for _, tid in stop_ids],
                seed=args.seed,
            )
        else:
            step_iter = generate(
                backend,
                args.prompt,
                sampler,
                max_tokens=args.max_tokens,
                top_k=args.top_k,
                rng=rng,
                stop_ids=[tid for _, tid in stop_ids],
            )

        for gs in step_iter:
            if first_token_at is None:
                first_token_at = _time.perf_counter()
            d = gs.decision
            chosen_cand = gs.chosen_candidate()
            chosen_is_special = chosen_cand.is_special if chosen_cand else False
            p_chosen = render.fmt_prob(chosen_cand.prob) if chosen_cand else "[dim]?[/dim]"
            if d.changed_greedy:
                greedy_text = next(
                    (c.text for c in gs.step_result.candidates if c.token_id == d.greedy_token_id),
                    "?",
                )
                vs = f"[yellow]!= {render.token_repr(greedy_text, 10)}[/yellow]"
            else:
                vs = "[green]= greedy[/green]"
            tops = "  ".join(
                f"{render.token_repr(c.text, 8, is_special=c.is_special)}={render.fmt_prob(c.prob)}"
                for c in gs.step_result.candidates[:5]
            )
            table.add_row(
                str(gs.step),
                render.token_repr(d.token_text, 14, is_special=chosen_is_special),
                p_chosen,
                vs,
                str(len(d.kept)),
                render.token_repr(d.note, 22),
                tops,
            )
            chosen_ids.append(d.token_id)
            last_stop_reason = gs.stop_reason
            last_chosen_text = d.token_text

        if timing is not None and first_token_at is not None:
            timing.record(
                "prompt eval + first token",
                first_token_at - loop_start,
                tokens=1,
            )
            timing.record(
                "decode",
                _time.perf_counter() - first_token_at,
                tokens=max(0, len(chosen_ids) - 1),
            )

        app.console.print(table)
        if last_stop_reason == "eos":
            chosen_last_id = chosen_ids[-1] if chosen_ids else -1
            label = render.token_repr(last_chosen_text, 24, is_special=True)
            app.console.print(
                f"[magenta]stopped on EOS[/magenta]: model emitted {label} (id={chosen_last_id})"
            )
        elif last_stop_reason == "user_stop":
            app.console.print(
                f"[dim]stopped on --stop token: {render.token_repr(last_chosen_text, 24)}[/dim]"
            )
        elif last_stop_reason == "max_tokens":
            app.console.print(
                f"[dim]reached --max-tokens={args.max_tokens} (model did not emit EOS).[/dim]"
            )
        completion = backend.detokenize(chosen_ids)
        app.console.print(f"\n[bold]prompt:[/bold] {args.prompt}")
        app.console.print(f"[bold]completion:[/bold][green]{completion}[/green]")
        if timing is not None:
            app.console.print(timing.render())
        return 0
    finally:
        if own_backend and backend is not None:
            backend.close()
