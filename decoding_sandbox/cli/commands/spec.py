"""``dsbx spec`` -- speculative decoding with accept/reject visualization."""

from __future__ import annotations

import argparse

from rich.table import Table

from decoding_sandbox.cli import app
from decoding_sandbox.cli._shared import _maybe_phase
from decoding_sandbox.cli.timing import Timing
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config


def cmd_spec(
    args: argparse.Namespace,
    cfg: Config,
    *,
    backend: Backend | None = None,  # accepted for signature symmetry; unused
    show_banner: bool = True,
) -> int:
    """Speculative decoding owns its own target + draft pair, even from
    inside ``session`` -- the session backend isn't necessarily the same as
    the speculative target. ``backend`` is accepted to keep the dispatcher's
    call signature uniform, but it's ignored.
    """
    del backend  # see docstring

    from decoding_sandbox.cli import render
    from decoding_sandbox.core.factory import build_backend
    from decoding_sandbox.core.speculative import speculative_generate

    rc = app._run_preflight(cfg, skip=getattr(args, "skip_preflight", False))
    if rc is not None:
        return rc

    timing = None if getattr(args, "no_timing", False) else Timing()
    with _maybe_phase(timing, "target load"):
        app.console.print(f"[dim]loading target '{args.target_model}'...[/dim]")
        target = build_backend("hf", cfg, model=args.target_model)
    with _maybe_phase(timing, "draft load"):
        app.console.print(f"[dim]loading draft '{args.draft_model}'...[/dim]")
        draft = build_backend("hf", cfg, model=args.draft_model)
    try:
        if show_banner:
            app.console.print(
                f"target: [cyan]{target.capabilities.name}[/cyan]  "
                f"draft: [magenta]{draft.capabilities.name}[/magenta]  "
                f"gamma={args.gamma}\n"
            )

        table = Table(title=f"speculative decode {args.prompt!r}")
        table.add_column("round", justify="right")
        table.add_column("draft proposed (green=accepted, red=rejected)")
        table.add_column("acc/total", justify="right")
        table.add_column("correction/bonus")

        total_proposed = total_accepted = rounds = total_emitted = 0
        all_ids: list[int] = []
        with _maybe_phase(timing, "speculative loop"):
            for rnd in speculative_generate(
                target,
                draft,
                args.prompt,
                gamma=args.gamma,
                max_tokens=args.max_tokens,
            ):
                parts = []
                for i, c in enumerate(rnd.proposed):
                    color = "green" if i < rnd.accepted else "red strike"
                    parts.append(f"[{color}]{render.token_repr(c.text, 10)}[/]")
                corr = render.token_repr(rnd.correction.text, 12) if rnd.correction else "-"
                table.add_row(
                    str(rnd.step),
                    " ".join(parts) or "[dim](none)[/dim]",
                    f"{rnd.accepted}/{len(rnd.proposed)}",
                    f"[cyan]{corr}[/cyan]",
                )
                total_proposed += len(rnd.proposed)
                total_accepted += rnd.accepted
                total_emitted += len(rnd.emitted_ids)
                rounds += 1
                all_ids.extend(rnd.emitted_ids)

        if timing is not None:
            timing.set_tokens("speculative loop", total_emitted)
        app.console.print(table)
        accept_rate = (total_accepted / total_proposed) if total_proposed else 0.0
        speedup = (total_emitted / rounds) if rounds else 0.0
        app.console.print(
            f"\nrounds={rounds}  tokens={total_emitted}  "
            f"draft acceptance={accept_rate:.1%}  "
            f"tokens/target-pass=[bold]{speedup:.2f}[/bold] "
            f"(plain greedy = 1.00; higher is faster)"
        )
        app.console.print(f"\n[bold]prompt:[/bold] {args.prompt}")
        app.console.print(f"[bold]completion:[/bold][green]{target.detokenize(all_ids)}[/green]")
        if timing is not None:
            app.console.print(timing.render())
        return 0
    finally:
        target.close()
        draft.close()
