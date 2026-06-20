"""CLI command dispatch.

Wave 0 implements `doctor` (environment + storage preflight) and `probe`
(provider logprob capability check). The decoding modes (`inspect`, `generate`,
`manual`) are wired as stubs and implemented in later waves.
"""

from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.table import Table

from decoding_sandbox import __version__
from decoding_sandbox.core import storage
from decoding_sandbox.core.config import Config, load_config

console = Console()


def _mask(secret: str | None) -> str:
    if not secret:
        return "[red]missing[/red]"
    if len(secret) <= 8:
        return "[green]set[/green] (****)"
    return f"[green]set[/green] ({secret[:4]}...{secret[-3:]})"


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
            _mask(prov.api_key()),
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


def cmd_inspect(args: argparse.Namespace, cfg: Config) -> int:
    from decoding_sandbox.cli import render
    from decoding_sandbox.core.factory import build_backend

    name = args.backend or cfg.default_backend
    console.print(f"[dim]building backend '{name}'...[/dim]")
    backend = build_backend(name, cfg, model=args.model)
    caps = backend.capabilities
    console.print(
        f"backend: [cyan]{caps.name}[/cyan]  "
        f"full_vocab={caps.full_vocab}  prompt_logprobs={caps.prompt_logprobs}  "
        f"max_top_logprobs={caps.max_top_logprobs}"
    )
    if caps.notes:
        console.print(f"[dim]{caps.notes}[/dim]")
    if not caps.full_vocab:
        console.print(
            "[dim]note: top-k backend -- a token's probability is shown only if it "
            "is within the returned top-k (others read '<top-k').[/dim]"
        )
    if not caps.full_vocab and not caps.prompt_logprobs:
        console.print(
            "[yellow]note: this backend has no native whole-context logprobs; "
            "each position is re-evaluated per prefix (slow, and chat-only providers "
            "are approximate).[/yellow]"
        )

    watch = _resolve_watch(backend, args.watch or [])
    generated_only_inspect = (
        backend.__class__.__name__ == "OpenAICompatBackend" and not caps.prompt_logprobs
    )
    if generated_only_inspect:
        console.print(
            "[yellow]note: this backend cannot score prompt tokens; showing the "
            "next-token distribution after the prompt instead.[/yellow]"
        )
        step = backend.next_distribution(backend.tokenize(args.prompt), top_k=args.top_k)
        step.context_text = args.prompt
        step.watched = {wid: backend._lookup_watch(step, wid) for _, wid in watch}
        steps = [step]
        title = f"Next-token inspection: {args.prompt!r}"
    else:
        steps = backend.score_prompt(args.prompt, top_k=args.top_k, watch_ids=[i for _, i in watch])
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
        nxt = render.token_repr(st.chosen.text, 14) if st.chosen else "?"
        p_next = render.fmt_prob(st.chosen.prob) if st.chosen else "?"
        rank = f"#{st.chosen.rank}" if (st.chosen and st.chosen.rank >= 0) else "[dim]?[/dim]"
        top = st.top
        conf = (
            f"{render.fmt_prob(st.confidence)} {render.token_repr(top.text, 12)!s}"
            if top else "-"
        )
        row = [str(st.position), f"{ctx} -> {nxt}", p_next, rank, conf]
        for _, wid in watch:
            row.append(render.watch_cell(st.watched.get(wid)))
        table.add_row(*row)

    console.print(table)

    if args.candidates:
        _print_candidates(steps, args.candidates, args.top_k)

    backend.close()
    return 0


def cmd_generate(args: argparse.Namespace, cfg: Config) -> int:
    import random

    from decoding_sandbox.cli import render
    from decoding_sandbox.core import samplers
    from decoding_sandbox.core.engine import generate
    from decoding_sandbox.core.factory import build_backend

    name = args.backend or cfg.default_backend
    console.print(f"[dim]building backend '{name}'...[/dim]")
    backend = build_backend(name, cfg, model=args.model)
    console.print(f"backend: [cyan]{backend.capabilities.name}[/cyan]")

    # Build the sampler (built-in or custom plug-in).
    if args.sampler == "custom":
        if not args.custom_file:
            console.print("[red]--sampler custom requires --custom-file path.py[:func][/red]")
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
    console.print(
        f"sampler: [magenta]{sampler_name}[/magenta]  seed={args.seed}  "
        f"max_tokens={args.max_tokens}\n"
    )

    table = Table(title=f"generate {args.prompt!r}")
    table.add_column("step", justify="right")
    table.add_column("chosen")
    table.add_column("p(chosen)", justify="right")
    table.add_column("vs greedy")
    table.add_column("kept", justify="right")
    table.add_column("sampler")
    table.add_column("top candidates")

    chosen_ids: list[int] = []
    for gs in generate(
        backend, args.prompt, sampler,
        max_tokens=args.max_tokens, top_k=args.top_k, rng=rng,
    ):
        d = gs.decision
        chosen_cand = gs.chosen_candidate()
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
            f"{render.token_repr(c.text, 8)}={render.fmt_prob(c.prob)}"
            for c in gs.step_result.candidates[:5]
        )
        table.add_row(
            str(gs.step),
            render.token_repr(d.token_text, 14),
            p_chosen,
            vs,
            str(len(d.kept)),
            render.token_repr(d.note, 22),
            tops,
        )
        chosen_ids.append(d.token_id)

    console.print(table)
    completion = backend.detokenize(chosen_ids)
    console.print(f"\n[bold]prompt:[/bold] {args.prompt}")
    console.print(f"[bold]completion:[/bold][green]{completion}[/green]")
    backend.close()
    return 0


def cmd_spec(args: argparse.Namespace, cfg: Config) -> int:
    from decoding_sandbox.cli import render
    from decoding_sandbox.core.factory import build_backend
    from decoding_sandbox.core.speculative import speculative_generate

    console.print(f"[dim]loading target '{args.target_model}'...[/dim]")
    target = build_backend("hf", cfg, model=args.target_model)
    console.print(f"[dim]loading draft '{args.draft_model}'...[/dim]")
    draft = build_backend("hf", cfg, model=args.draft_model)
    console.print(
        f"target: [cyan]{target.capabilities.name}[/cyan]  "
        f"draft: [magenta]{draft.capabilities.name}[/magenta]  gamma={args.gamma}\n"
    )

    table = Table(title=f"speculative decode {args.prompt!r}")
    table.add_column("round", justify="right")
    table.add_column("draft proposed (green=accepted, red=rejected)")
    table.add_column("acc/total", justify="right")
    table.add_column("correction/bonus")

    total_proposed = total_accepted = rounds = total_emitted = 0
    all_ids: list[int] = []
    for rnd in speculative_generate(
        target, draft, args.prompt, gamma=args.gamma, max_tokens=args.max_tokens
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
    console.print(f"[bold]completion:[/bold][green]{target.detokenize(all_ids)}[/green]")
    target.close()
    draft.close()
    return 0


def cmd_manual(args: argparse.Namespace, cfg: Config) -> int:
    from decoding_sandbox.cli.manual_tui import run_manual
    from decoding_sandbox.core.factory import build_backend

    name = args.backend or cfg.default_backend
    console.print(f"[dim]building backend '{name}'...[/dim]")
    backend = build_backend(name, cfg, model=args.model)
    return run_manual(backend, args.prompt, top_k=args.top_k)


def _print_candidates(steps, max_positions: int, top_k: int) -> None:
    from decoding_sandbox.cli import render

    console.print(f"\n[bold]Top-{top_k} candidates (first {max_positions} positions)[/bold]")
    for st in steps[:max_positions]:
        ctx = render.token_repr(st.context_text or "", 14)
        line = f"[cyan]pos {st.position}[/cyan] after {ctx!r}: "
        line += "  ".join(
            f"{render.token_repr(c.text, 10)!s}={render.fmt_prob(c.prob)}"
            for c in st.candidates
        )
        console.print(line)


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
    p_inspect.add_argument(
        "--backend", default=None, help="hf | llamacpp (default: config run.backend)."
    )
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
    p_inspect.set_defaults(func=cmd_inspect)

    p_gen = sub.add_parser("generate", help="Decode with a chosen/custom sampler, step by step.")
    p_gen.add_argument("prompt", help="Text to continue.")
    p_gen.add_argument("--backend", default=None, help="hf | llamacpp.")
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
    p_gen.set_defaults(func=cmd_generate)

    p_manual = sub.add_parser("manual", help="Interactive token-by-token decoding (TUI).")
    p_manual.add_argument("prompt", help="Starting text.")
    p_manual.add_argument("--backend", default=None, help="hf | llamacpp.")
    p_manual.add_argument("--model", default=None, help="Override the model id.")
    p_manual.add_argument("--top-k", type=int, default=12, help="Candidates shown per step.")
    p_manual.set_defaults(func=cmd_manual)

    p_spec = sub.add_parser(
        "spec", help="Speculative decoding (HF draft+target) with accept/reject view."
    )
    p_spec.add_argument("prompt", help="Text to continue.")
    p_spec.add_argument("--target-model", default="Qwen/Qwen3-1.7B-Base")
    p_spec.add_argument("--draft-model", default="Qwen/Qwen3-0.6B-Base")
    p_spec.add_argument("--gamma", type=int, default=4, help="Draft tokens proposed per round.")
    p_spec.add_argument("--max-tokens", type=int, default=24)
    p_spec.set_defaults(func=cmd_spec)

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
