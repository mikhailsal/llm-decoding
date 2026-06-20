"""CLI command dispatch.

Subcommands (all backed by ``decoding_sandbox.core``):
- ``doctor``  : environment + provider keys + disk free-space report
- ``probe``   : live provider logprob capability check
- ``inspect`` : per-token confidence + watch-token highlighting
- ``generate``: decode with a chosen/custom sampler, per-step diff vs greedy
- ``manual``  : interactive token-by-token TUI (prompt_toolkit)
- ``spec``    : speculative decoding with accept/reject visualization
- ``session`` : long-lived REPL that keeps the model loaded across commands

Every heavy command runs ``storage.preflight_or_raise`` first; pass
``--skip-preflight`` to bypass. Every heavy command also prints a one-line
timing summary (``timing: prompt eval ... | total ...``); suppress with
``--no-timing`` (or ``:timing off`` inside ``session``).
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager

from rich.console import Console
from rich.table import Table

from decoding_sandbox import __version__
from decoding_sandbox.cli.timing import Timing
from decoding_sandbox.core import storage
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config, load_config

console = Console()

# Providers that legitimately do not need an API key (e.g. local OpenAI-compat
# servers). Doctor shows "no key needed" instead of red "missing" for these.
_NO_KEY_PROVIDERS = frozenset({"lmstudio"})


def _mask(secret: str | None, *, no_key_ok: bool = False) -> str:
    if not secret:
        if no_key_ok:
            return "[dim]no key needed[/dim]"
        return "[red]missing[/red]"
    if len(secret) <= 8:
        return "[green]set[/green] (****)"
    return f"[green]set[/green] ({secret[:4]}...{secret[-3:]})"


def _run_preflight(cfg: Config, *, skip: bool) -> int | None:
    """Abort the current command if disk free space is below the floor.

    Returns ``None`` on success, or an exit code (>0) on failure. ``skip=True``
    short-circuits to ``None`` so users can override the check if they know
    what they're doing.
    """
    if skip:
        return None
    try:
        storage.preflight_or_raise(cfg.storage.check_paths, cfg.storage.min_free_gb)
    except storage.StoragePreflightError as exc:
        console.print(f"[red]preflight failed:[/red] {exc}")
        console.print(
            "[dim]pass --skip-preflight to bypass this check.[/dim]"
        )
        return 3
    return None


def cmd_doctor(args: argparse.Namespace, cfg: Config) -> int:
    console.rule(f"[bold]Decoding Sandbox doctor[/bold] v{__version__}")

    # Config source / secrets
    console.print(
        f"Config file : [cyan]{cfg.config_path or 'built-in defaults'}[/cyan]"
    )
    console.print(
        f"Secrets file: [cyan]{cfg.secrets_env_file or '(none)'}[/cyan]"
    )
    console.print(f"Host        : [cyan]{os.uname().nodename}[/cyan]\n")

    # Provider keys
    ptable = Table(title="Provider API keys", show_edge=False)
    ptable.add_column("provider")
    ptable.add_column("env var")
    ptable.add_column("status")
    ptable.add_column("prompt-logprobs")
    ptable.add_column("max top_logprobs")
    for name, prov in sorted(cfg.providers.items()):
        ptable.add_row(
            name,
            prov.api_key_env,
            _mask(prov.api_key(), no_key_ok=name in _NO_KEY_PROVIDERS),
            "yes" if prov.supports_prompt_logprobs else "no",
            str(prov.max_top_logprobs),
        )
    console.print(ptable)

    # Storage preflight (display all, flag low ones)
    statuses = storage.check_paths(cfg.storage.check_paths, cfg.storage.min_free_gb)
    stable = Table(title=f"Storage (floor {cfg.storage.min_free_gb} GB)", show_edge=False)
    stable.add_column("path")
    stable.add_column("status")
    stable.add_column("free GB", justify="right")
    stable.add_column("total GB", justify="right")
    any_low = False
    for s in statuses:
        if not s.exists:
            stable.add_row(s.path, "[dim]skipped[/dim]", "-", "-")
            continue
        status = "[green]OK[/green]" if s.ok else "[red]LOW[/red]"
        any_low = any_low or not s.ok
        stable.add_row(s.path, status, f"{s.free_gb:.1f}", f"{s.total_gb:.1f}")
    console.print(stable)

    # Optional local-engine availability (only meaningful on dsbx-host)
    console.print()
    _report_local_engines()

    if any_low:
        console.print(
            "\n[red]Warning:[/red] at least one disk is below the free-space floor."
        )
        return 1
    console.print("\n[green]All checks passed.[/green]")
    return 0


def _report_local_engines() -> None:
    table = Table(title="Local engines", show_edge=False)
    table.add_column("component")
    table.add_column("status")

    try:
        import torch  # type: ignore

        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "cpu only"
        table.add_row("torch", f"[green]{torch.__version__}[/green] (cuda={cuda}, {dev})")
    except Exception as exc:  # noqa: BLE001
        table.add_row("torch", f"[yellow]not installed[/yellow] ({type(exc).__name__})")

    try:
        import transformers  # type: ignore

        table.add_row("transformers", f"[green]{transformers.__version__}[/green]")
    except Exception:  # noqa: BLE001
        table.add_row("transformers", "[yellow]not installed[/yellow]")

    try:
        import bitsandbytes  # type: ignore

        table.add_row("bitsandbytes", f"[green]{bitsandbytes.__version__}[/green]")
    except Exception:  # noqa: BLE001
        table.add_row("bitsandbytes", "[yellow]not installed[/yellow]")

    try:
        import llama_cpp  # type: ignore

        ver = getattr(llama_cpp, "__version__", "?")
        # Probing CUDA support without loading a model is hard; just report the
        # binding version. The build flags went through CMAKE_ARGS at install.
        table.add_row(
            "llama-cpp-python", f"[green]{ver}[/green] (for the llamacpp-py backend)"
        )
    except Exception:  # noqa: BLE001
        table.add_row(
            "llama-cpp-python",
            "[yellow]not installed[/yellow] (needed for llamacpp-py)",
        )

    console.print(table)


def cmd_probe(args: argparse.Namespace, cfg: Config) -> int:
    from decoding_sandbox.core import provider_probe

    return provider_probe.run_probe(
        cfg,
        providers=args.providers,
        model=args.model,
        console=console,
    )


def _resolve_watch(backend, watch: list[str]) -> list[tuple[str, int]]:
    """Map each watch string to a single token id (warn if multi-token)."""
    resolved: list[tuple[str, int]] = []
    for w in watch:
        ids = backend.tokenize(w)
        if not ids:
            console.print(f"[yellow]watch {w!r}: tokenizes to nothing, skipped[/yellow]")
            continue
        if len(ids) > 1:
            console.print(
                f"[yellow]watch {w!r}: {len(ids)} tokens; watching first "
                f"({backend.piece(ids[0])!r}). Try a leading space.[/yellow]"
            )
        resolved.append((w, ids[0]))
    return resolved


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
            console.print(f"[yellow]stop {s!r}: tokenizes to nothing, skipped[/yellow]")
            continue
        if len(ids) > 1:
            console.print(
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
    from decoding_sandbox.core.factory import build_backend

    console.print(f"[dim]building backend '{name}'...[/dim]")
    if timing is None:
        return build_backend(name, cfg, model=model)
    with timing.phase("backend load"):
        return build_backend(name, cfg, model=model)


def _print_backend_banner(backend: Backend, *, out: Console | None = None) -> None:
    """Print the capability banner that ``inspect`` historically led with.

    Also surfaces the same backend-specific notes (no native whole-context,
    top-k only, llamacpp HTTP nudge to llamacpp-py). ``out`` defaults to the
    module-level rich console; the session REPL passes its own buffered
    console so meta commands write where the caller expects.
    """
    out = out or console
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
        from decoding_sandbox.cli import render as _render

        pieces: list[str] = []
        for tid in caps.eos_token_ids:
            text = backend.piece(tid) if hasattr(backend, "piece") else ""
            pieces.append(
                f"{tid}={_render.token_repr(text, 16, is_special=True)}"
            )
        out.print(f"[dim]EOS ids: {', '.join(pieces)}[/dim]")
    else:
        out.print(
            "[dim]EOS ids: <not exposed by this backend>[/dim]"
        )
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
        rc = _run_preflight(cfg, skip=getattr(args, "skip_preflight", False))
        if rc is not None:
            return rc

    timing = None if getattr(args, "no_timing", False) else Timing()
    try:
        if own_backend:
            name = args.backend or cfg.default_backend
            backend = _build_backend_with_load_timing(
                name, cfg, model=args.model, timing=timing
            )
        if show_banner:
            _print_backend_banner(backend)

        watch = _resolve_watch(backend, args.watch or [])
        caps = backend.capabilities
        prompt_tokens = backend.tokenize(args.prompt)
        generated_only_inspect = (
            backend.__class__.__name__ == "OpenAICompatBackend"
            and not caps.prompt_logprobs
        )
        if generated_only_inspect:
            console.print(
                "[yellow]note: this backend cannot score prompt tokens; showing the "
                "next-token distribution after the prompt instead.[/yellow]"
            )
            tok_count = 1
            with _maybe_phase(timing, "next-token distribution", tokens=tok_count):
                step = backend.next_distribution(prompt_tokens, top_k=args.top_k)
            step.context_text = args.prompt
            step.watched = {wid: backend.lookup_watch(step, wid) for _, wid in watch}
            steps = [step]
            title = f"Next-token inspection: {args.prompt!r}"
        else:
            tok_count = len(prompt_tokens)
            with _maybe_phase(timing, "score_prompt", tokens=tok_count):
                steps = backend.score_prompt(
                    args.prompt,
                    top_k=args.top_k,
                    watch_ids=[i for _, i in watch],
                )
            title = f"Context inspection: {args.prompt!r}"

        table = Table(title=title)
        table.add_column("pos", justify="right")
        table.add_column("context -> next")
        table.add_column("p(next)", justify="right")
        table.add_column("rank", justify="right")
        table.add_column("confidence (top-1)")
        for w, _ in watch:
            table.add_column(f"watch {w!r}")

        for st in steps:
            ctx = render.token_repr(st.context_text or "", 14)
            if st.chosen is not None:
                nxt = render.token_repr(
                    st.chosen.text, 14, is_special=st.chosen.is_special
                )
            else:
                nxt = "?"
            p_next = render.fmt_prob(st.chosen.prob) if st.chosen else "?"
            rank = (
                f"#{st.chosen.rank}"
                if (st.chosen and st.chosen.rank >= 0)
                else "[dim]?[/dim]"
            )
            top = st.top
            conf = (
                f"{render.fmt_prob(st.confidence)} "
                f"{render.token_repr(top.text, 12, is_special=top.is_special)!s}"
                if top
                else "-"
            )
            row = [str(st.position), f"{ctx} -> {nxt}", p_next, rank, conf]
            for _, wid in watch:
                row.append(render.watch_cell(st.watched.get(wid)))
            table.add_row(*row)

        console.print(table)

        if args.candidates:
            _print_candidates(steps, args.candidates, args.top_k)

        if timing is not None:
            console.print(timing.render())
        return 0
    finally:
        if own_backend and backend is not None:
            backend.close()


@contextmanager
def _null_phase():
    """No-op context manager used when timing is disabled."""
    yield


def _maybe_phase(timing: Timing | None, name: str, *, tokens: int | None = None):
    """Return ``timing.phase(...)`` or a no-op context manager when None."""
    if timing is not None:
        return timing.phase(name, tokens=tokens)
    return _null_phase()


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
        rc = _run_preflight(cfg, skip=getattr(args, "skip_preflight", False))
        if rc is not None:
            return rc

    timing = None if getattr(args, "no_timing", False) else Timing()
    try:
        if own_backend:
            name = args.backend or cfg.default_backend
            backend = _build_backend_with_load_timing(
                name, cfg, model=args.model, timing=timing
            )
        if show_banner:
            console.print(f"backend: [cyan]{backend.capabilities.name}[/cyan]")

        # Sampler construction (built-in or custom plug-in).
        if args.sampler == "custom":
            if not args.custom_file:
                console.print(
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
        console.print(
            f"sampler: [magenta]{sampler_name}[/magenta]  seed={args.seed}  "
            f"max_tokens={args.max_tokens}"
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
        # The split happens at the first ``gs`` yielded by ``generate``.
        import time as _time

        chosen_ids: list[int] = []
        loop_start = _time.perf_counter()
        first_token_at: float | None = None
        last_stop_reason: str | None = None
        last_chosen_text: str = ""

        for gs in generate(
            backend, args.prompt, sampler,
            max_tokens=args.max_tokens, top_k=args.top_k, rng=rng,
            stop_ids=[tid for _, tid in stop_ids],
        ):
            if first_token_at is None:
                first_token_at = _time.perf_counter()
            d = gs.decision
            chosen_cand = gs.chosen_candidate()
            chosen_is_special = chosen_cand.is_special if chosen_cand else False
            p_chosen = (
                render.fmt_prob(chosen_cand.prob)
                if chosen_cand
                else "[dim]?[/dim]"
            )
            if d.changed_greedy:
                greedy_text = next(
                    (c.text for c in gs.step_result.candidates
                     if c.token_id == d.greedy_token_id),
                    "?",
                )
                vs = f"[yellow]!= {render.token_repr(greedy_text, 10)}[/yellow]"
            else:
                vs = "[green]= greedy[/green]"
            tops = "  ".join(
                f"{render.token_repr(c.text, 8, is_special=c.is_special)}="
                f"{render.fmt_prob(c.prob)}"
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

        console.print(table)
        if last_stop_reason == "eos":
            chosen_last_id = chosen_ids[-1] if chosen_ids else -1
            label = render.token_repr(last_chosen_text, 24, is_special=True)
            console.print(
                f"[magenta]stopped on EOS[/magenta]: model emitted {label} "
                f"(id={chosen_last_id})"
            )
        elif last_stop_reason == "user_stop":
            console.print(
                f"[dim]stopped on --stop token: "
                f"{render.token_repr(last_chosen_text, 24)}[/dim]"
            )
        elif last_stop_reason == "max_tokens":
            console.print(
                f"[dim]reached --max-tokens={args.max_tokens} "
                "(model did not emit EOS).[/dim]"
            )
        completion = backend.detokenize(chosen_ids)
        console.print(f"\n[bold]prompt:[/bold] {args.prompt}")
        console.print(f"[bold]completion:[/bold][green]{completion}[/green]")
        if timing is not None:
            console.print(timing.render())
        return 0
    finally:
        if own_backend and backend is not None:
            backend.close()


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

    rc = _run_preflight(cfg, skip=getattr(args, "skip_preflight", False))
    if rc is not None:
        return rc

    timing = None if getattr(args, "no_timing", False) else Timing()
    with _maybe_phase(timing, "target load"):
        console.print(f"[dim]loading target '{args.target_model}'...[/dim]")
        target = build_backend("hf", cfg, model=args.target_model)
    with _maybe_phase(timing, "draft load"):
        console.print(f"[dim]loading draft '{args.draft_model}'...[/dim]")
        draft = build_backend("hf", cfg, model=args.draft_model)
    try:
        if show_banner:
            console.print(
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
                target, draft, args.prompt,
                gamma=args.gamma, max_tokens=args.max_tokens,
            ):
                parts = []
                for i, c in enumerate(rnd.proposed):
                    color = "green" if i < rnd.accepted else "red strike"
                    parts.append(f"[{color}]{render.token_repr(c.text, 10)}[/]")
                corr = (
                    render.token_repr(rnd.correction.text, 12)
                    if rnd.correction else "-"
                )
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
        console.print(table)
        accept_rate = (total_accepted / total_proposed) if total_proposed else 0.0
        speedup = (total_emitted / rounds) if rounds else 0.0
        console.print(
            f"\nrounds={rounds}  tokens={total_emitted}  "
            f"draft acceptance={accept_rate:.1%}  "
            f"tokens/target-pass=[bold]{speedup:.2f}[/bold] "
            f"(plain greedy = 1.00; higher is faster)"
        )
        console.print(f"\n[bold]prompt:[/bold] {args.prompt}")
        console.print(
            f"[bold]completion:[/bold][green]{target.detokenize(all_ids)}[/green]"
        )
        if timing is not None:
            console.print(timing.render())
        return 0
    finally:
        target.close()
        draft.close()


def cmd_manual(
    args: argparse.Namespace,
    cfg: Config,
    *,
    backend: Backend | None = None,
    show_banner: bool = True,
) -> int:
    del show_banner  # the manual TUI prints its own header
    from decoding_sandbox.cli.manual_tui import run_manual

    own_backend = backend is None
    if own_backend:
        rc = _run_preflight(cfg, skip=getattr(args, "skip_preflight", False))
        if rc is not None:
            return rc
        name = args.backend or cfg.default_backend
        backend = _build_backend_with_load_timing(
            name, cfg, model=args.model, timing=None
        )
    return run_manual(
        backend, args.prompt, top_k=args.top_k, own_backend=own_backend
    )


def cmd_session(args: argparse.Namespace, cfg: Config) -> int:
    """Long-lived REPL that keeps a backend loaded across commands.

    The heavy load (e.g. 30s+ for the 9B GGUF) happens once at startup; every
    subsequent ``inspect``/``generate``/``manual`` runs in-process and skips
    the load. The session also runs the disk preflight once up front (the
    in-REPL commands inherit ``skip_preflight=True`` from the session parser).
    """
    from decoding_sandbox.cli.session import (
        SessionState,
        run_session,
    )

    rc = _run_preflight(cfg, skip=getattr(args, "skip_preflight", False))
    if rc is not None:
        return rc

    name = args.backend or cfg.default_backend
    timing = None if getattr(args, "no_timing", False) else Timing()
    backend = _build_backend_with_load_timing(
        name, cfg, model=args.model, timing=timing
    )
    _print_backend_banner(backend)
    if timing is not None:
        console.print(timing.render(prefix="startup"))

    state = SessionState(
        cfg=cfg,
        backend=backend,
        backend_name=name,
        backend_model=args.model,
        console=console,
        timing_enabled=not getattr(args, "no_timing", False),
    )
    try:
        return run_session(state)
    finally:
        try:
            state.backend.close()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass


def _print_candidates(steps, max_positions: int, top_k: int) -> None:
    from decoding_sandbox.cli import render

    console.print(f"\n[bold]Top-{top_k} candidates (first {max_positions} positions)[/bold]")
    for st in steps[:max_positions]:
        ctx = render.token_repr(st.context_text or "", 14)
        line = f"[cyan]pos {st.position}[/cyan] after {ctx!r}: "
        line += "  ".join(
            f"{render.token_repr(c.text, 10, is_special=c.is_special)!s}="
            f"{render.fmt_prob(c.prob)}"
            for c in st.candidates
        )
        console.print(line)


_BACKEND_HELP = (
    "Backend name: built-ins are 'hf' (HF transformers full-vocab), "
    "'llamacpp' (HTTP top-k via llama-server), and 'llamacpp-py' "
    "(in-process llama-cpp-python with FULL vocab via logits_all=True -- "
    "white-box for GGUFs HF can't load); any provider configured in "
    "config.toml (e.g. fireworks, nim, openrouter, lmstudio) also works. "
    "Default: config run.backend."
)


def _add_preflight_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the disk free-space check before running this command.",
    )


def _add_timing_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--no-timing",
        action="store_true",
        help=(
            "Suppress the one-line timing summary printed after the command "
            "(prompt eval / decode / total wall time + tokens-per-second)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsbx",
        description="Decoding Sandbox -- study LLM token probabilities and decoding.",
    )
    parser.add_argument("--config", help="Path to a config.toml (overrides discovery).")
    parser.add_argument("--version", action="version", version=f"dsbx {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Check environment, keys, and disk space.")
    p_doctor.set_defaults(func=cmd_doctor)

    p_probe = sub.add_parser("probe", help="Live-check provider logprob capabilities.")
    p_probe.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Subset of providers to probe (default: all configured).",
    )
    p_probe.add_argument("--model", default=None, help="Override the model to probe.")
    p_probe.set_defaults(func=cmd_probe)

    p_inspect = sub.add_parser(
        "inspect", help="Per-token confidence + watch-token highlighting for a prompt."
    )
    p_inspect.add_argument("prompt", help="Text to inspect.")
    p_inspect.add_argument("--backend", default=None, help=_BACKEND_HELP)
    p_inspect.add_argument("--model", default=None, help="Override the model id.")
    p_inspect.add_argument("--top-k", type=int, default=8, help="Candidates per position.")
    p_inspect.add_argument(
        "--watch", action="append", default=[],
        help="Token text to highlight at every position (repeatable). Use a leading space, e.g. --watch ' Paris'.",
    )
    p_inspect.add_argument(
        "--candidates", type=int, default=0, metavar="N",
        help="Also print the full top-k candidate list for the first N positions.",
    )
    _add_preflight_flag(p_inspect)
    _add_timing_flag(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    p_gen = sub.add_parser("generate", help="Decode with a chosen/custom sampler, step by step.")
    p_gen.add_argument("prompt", help="Text to continue.")
    p_gen.add_argument("--backend", default=None, help=_BACKEND_HELP)
    p_gen.add_argument("--model", default=None, help="Override the model id.")
    p_gen.add_argument(
        "--sampler", default="greedy",
        choices=["greedy", "temperature", "top_k", "top_p", "min_p", "typical", "custom"],
        help="Decoding function.",
    )
    p_gen.add_argument("--custom-file", default=None, help="path.py[:func] for --sampler custom.")
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--sampler-top-k", type=int, default=None, help="top_k for the top_k sampler.")
    p_gen.add_argument("--top-p", type=float, default=None)
    p_gen.add_argument("--min-p", type=float, default=None)
    p_gen.add_argument("--typical-p", type=float, default=None)
    p_gen.add_argument("--max-tokens", type=int, default=20)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument(
        "--top-k", type=int, default=50,
        help="How many candidates to pull from the backend per step (sampler input).",
    )
    p_gen.add_argument(
        "--stop", action="append", default=[],
        help=(
            "Stop generation as soon as this single-token string is chosen "
            "(repeatable). Multi-token strings are warned-about and ignored."
        ),
    )
    _add_preflight_flag(p_gen)
    _add_timing_flag(p_gen)
    p_gen.set_defaults(func=cmd_generate)

    p_manual = sub.add_parser("manual", help="Interactive token-by-token decoding (TUI).")
    p_manual.add_argument("prompt", help="Starting text.")
    p_manual.add_argument("--backend", default=None, help=_BACKEND_HELP)
    p_manual.add_argument("--model", default=None, help="Override the model id.")
    p_manual.add_argument("--top-k", type=int, default=12, help="Candidates shown per step.")
    _add_preflight_flag(p_manual)
    p_manual.set_defaults(func=cmd_manual)

    p_spec = sub.add_parser(
        "spec", help="Speculative decoding (HF draft+target) with accept/reject view."
    )
    p_spec.add_argument("prompt", help="Text to continue.")
    p_spec.add_argument("--target-model", default="Qwen/Qwen3-1.7B-Base")
    p_spec.add_argument("--draft-model", default="Qwen/Qwen3-0.6B-Base")
    p_spec.add_argument("--gamma", type=int, default=4, help="Draft tokens proposed per round.")
    p_spec.add_argument("--max-tokens", type=int, default=24)
    _add_preflight_flag(p_spec)
    _add_timing_flag(p_spec)
    p_spec.set_defaults(func=cmd_spec)

    p_session = sub.add_parser(
        "session",
        help=(
            "Long-lived REPL that keeps the model loaded across commands. "
            "Amortizes the slow GGUF/HF load over many inspects/generates."
        ),
    )
    p_session.add_argument("--backend", default=None, help=_BACKEND_HELP)
    p_session.add_argument("--model", default=None, help="Override the model id.")
    _add_preflight_flag(p_session)
    _add_timing_flag(p_session)
    p_session.set_defaults(func=cmd_session)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:  # pragma: no cover
        console.print("\n[dim]interrupted[/dim]")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
