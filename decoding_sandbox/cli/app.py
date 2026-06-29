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
import sys

from rich.console import Console

from decoding_sandbox import __version__
from decoding_sandbox.core import storage
from decoding_sandbox.core.config import Config, load_config


def _make_console(mode: str = "auto") -> Console:
    """Build a rich Console honoring an explicit color mode.

    Rich's default ``Console()`` calls ``sys.stdout.isatty()`` and disables
    ANSI when it's False -- the right thing for a real pipe-to-file, but
    the wrong thing for the common ``ssh dsbx-host 'dsbx inspect ...'``
    workflow: the user wants colour in their terminal, but stdout isn't a
    TTY on the remote side, so rich silently strips every ``[green]...``
    tag and the whole confidence/special-token visual encoding disappears.

    Three modes (matches ``ls``/``grep``/``git`` conventions):

    * ``"auto"``  -- rich's default detection. Colour when stdout is a
      TTY, plain otherwise. ``FORCE_COLOR=1`` / ``NO_COLOR=1`` env vars
      still apply via rich's own logic.
    * ``"always"`` -- force ANSI emission regardless of TTY detection.
      Useful over non-interactive SSH or when capturing for paste into a
      colour-capable terminal.
    * ``"never"``  -- disable colour even when stdout is a TTY (some
      legacy log scrapers can't strip ANSI).
    """
    mode = (mode or "auto").lower()
    if mode == "always":
        return Console(force_terminal=True, color_system="truecolor")
    if mode == "never":
        return Console(no_color=True)
    if mode != "auto":  # defensive -- argparse choices should prevent this
        mode = "auto"
    return Console()


console = _make_console("auto")


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
        console.print("[dim]pass --skip-preflight to bypass this check.[/dim]")
        return 3
    return None


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
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "Color rendering mode. 'auto' (default) emits ANSI when stdout "
            "is a terminal, plain otherwise -- which strips all rich "
            "highlighting under non-interactive SSH ('ssh dsbx-host dsbx ...'). "
            "Use 'always' to force colour over SSH (you can also set "
            "FORCE_COLOR=1); use 'never' to disable colour even on a TTY."
        ),
    )
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
        "--watch",
        action="append",
        default=[],
        help=(
            "Token text to highlight at every position (repeatable). "
            "Use a leading space, e.g. --watch ' Paris'."
        ),
    )
    p_inspect.add_argument(
        "--watch-id",
        action="append",
        type=int,
        default=[],
        metavar="N",
        help=(
            "Watch a specific token id (repeatable). Bypasses the text -> id "
            "round-trip, so it works for reserved/control tokens whose "
            "detokenized piece is empty or unprintable (EOS/BOS/PAD/<|...|>)."
        ),
    )
    p_inspect.add_argument(
        "--watch-eos",
        action="store_true",
        default=False,
        help=(
            "Convenience: expand to one watch column per id in "
            "backend.capabilities.eos_token_ids. Use this to track how the "
            "model's probability for EOS evolves across a fixed context."
        ),
    )
    p_inspect.add_argument(
        "--candidates",
        type=int,
        default=0,
        metavar="N",
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
        "--sampler",
        default="greedy",
        choices=["greedy", "temperature", "top_k", "top_p", "min_p", "typical", "custom"],
        help="Decoding function.",
    )
    p_gen.add_argument("--custom-file", default=None, help="path.py[:func] for --sampler custom.")
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument(
        "--sampler-top-k", type=int, default=None, help="top_k for the top_k sampler."
    )
    p_gen.add_argument("--top-p", type=float, default=None)
    p_gen.add_argument("--min-p", type=float, default=None)
    p_gen.add_argument("--typical-p", type=float, default=None)
    p_gen.add_argument("--max-tokens", type=int, default=20)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="How many candidates to pull from the backend per step (sampler input).",
    )
    p_gen.add_argument(
        "--stop",
        action="append",
        default=[],
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

    p_serve = sub.add_parser(
        "serve",
        help=(
            "Run the dsbx HTTP server (FastAPI + uvicorn) wrapping one heavy "
            "in-process backend. Clients connect via the 'remote' backend or a "
            "[remote.NAME] alias. Requires the [server] extra."
        ),
    )
    p_serve.add_argument(
        "--backend",
        choices=("hf", "llamacpp-py"),
        required=True,
        help="Which in-process backend to host (heavy local engines only).",
    )
    p_serve.add_argument("--model", default=None, help="Override the model id / GGUF path.")
    p_serve.add_argument(
        "--no-preload",
        action="store_true",
        help=(
            "Start with no model loaded (empty slot). Load one on demand via "
            "POST /v1/reload or the web UI's Status page model control. "
            "Useful when you want to pick the model from the browser."
        ),
    )
    p_serve.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address. Default is loopback (only this machine). Use a LAN "
            "address (or 0.0.0.0) to let the client client reach the server; "
            "a warning is printed because there is no auth."
        ),
    )
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="uvicorn log verbosity.",
    )
    p_serve.set_defaults(func=cmd_serve)

    p_web = sub.add_parser(
        "web",
        help=(
            "Run the dsbx web middleware (FastAPI + uvicorn) -- the browser-"
            "facing API that hides every backend key and URL behind one bearer "
            "token. Requires the [web] extra."
        ),
    )
    p_web.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address. Default is loopback. The web middleware "
            "authenticates every request, but pick the bind address with "
            "intent if you're putting it on a LAN interface."
        ),
    )
    p_web.add_argument("--port", type=int, default=8765)
    p_web.add_argument(
        "--token",
        default=None,
        help=(
            "Bearer token the browser must send. Defaults to $DSBX_WEB_TOKEN "
            "and then to [web].api_token in config.toml."
        ),
    )
    p_web.add_argument(
        "--frontend-dist",
        default=None,
        help=(
            "Path to a built SvelteKit bundle to static-serve at /. If omitted, "
            "only the JSON API is exposed (e.g. for dev where the frontend is "
            "served by `pnpm dev` on a different origin)."
        ),
    )
    p_web.add_argument(
        "--server-label",
        default="dsbx-web",
        help=(
            "Cosmetic label echoed by /api/v1/health and /api/v1/info so an "
            "operator can tell instances apart in a screenshot."
        ),
    )
    p_web.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="uvicorn log verbosity.",
    )
    p_web.set_defaults(func=cmd_web)

    p_session = sub.add_parser(
        "session",
        help=(
            "Convenience REPL with command history and a single loaded "
            "backend. Useful for fast iteration; for amortizing the slow "
            "GGUF/HF load across machines/processes, run `dsbx serve` on "
            "dsbx-host and use a [remote.NAME] backend instead."
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
    # Rebuild the module-level console only when the user explicitly opted
    # in to a non-auto colour mode. "auto" is rich's normal TTY detection,
    # which is already what's in place from module load -- and which the
    # captured_console test fixture monkeypatches before invoking main(),
    # so reassigning unconditionally would defeat the patch.
    color_mode = getattr(args, "color", "auto")
    if color_mode != "auto":
        global console
        console = _make_console(color_mode)
    cfg = load_config(args.config)
    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:  # pragma: no cover
        console.print("\n[dim]interrupted[/dim]")
        return 130
    except Exception as exc:
        # RemoteBackendError (and a few other "network failed / config
        # wrong" errors) are routine for a tool that talks to a server on
        # another host -- not programming bugs. Render them as one clean
        # red line + exit 4 instead of dumping a stack trace. Importing
        # the class lazily keeps the CLI usable even when the [server]
        # extra isn't installed (RemoteBackend lives in backends/, not
        # server/, but the safety net stays the same).
        from decoding_sandbox.backends.remote import RemoteBackendError

        if isinstance(exc, RemoteBackendError):
            console.print(f"[red]remote backend error:[/red] {exc}")
            console.print(
                "[dim]tip: run [bold]dsbx doctor[/bold] to probe each "
                r"configured \[remote.NAME] server, or check that "
                "[bold]dsbx serve[/bold] is running on the host.[/dim]"
            )
            return 4
        raise


# Re-export the subcommand handlers and shared helpers so existing call sites
# and tests that reference ``decoding_sandbox.cli.app.<name>`` keep working
# after the split into ``commands/`` and ``_shared``. The ``x as x`` form marks
# these as intentional re-exports (so the linter doesn't prune them).
from decoding_sandbox.cli._shared import WatchTarget as WatchTarget  # noqa: E402
from decoding_sandbox.cli._shared import (  # noqa: E402
    _build_backend_with_load_timing as _build_backend_with_load_timing,
)
from decoding_sandbox.cli._shared import (  # noqa: E402
    _collect_watch_targets as _collect_watch_targets,
)
from decoding_sandbox.cli._shared import _maybe_phase as _maybe_phase  # noqa: E402
from decoding_sandbox.cli._shared import _null_phase as _null_phase  # noqa: E402
from decoding_sandbox.cli._shared import (  # noqa: E402
    _print_backend_banner as _print_backend_banner,
)
from decoding_sandbox.cli._shared import _print_candidates as _print_candidates  # noqa: E402
from decoding_sandbox.cli._shared import _resolve_stop_ids as _resolve_stop_ids  # noqa: E402
from decoding_sandbox.cli._shared import _resolve_watch as _resolve_watch  # noqa: E402
from decoding_sandbox.cli._shared import _resolve_watch_eos as _resolve_watch_eos  # noqa: E402
from decoding_sandbox.cli._shared import _resolve_watch_ids as _resolve_watch_ids  # noqa: E402
from decoding_sandbox.cli.commands.doctor import (  # noqa: E402
    _NO_KEY_PROVIDERS as _NO_KEY_PROVIDERS,
)
from decoding_sandbox.cli.commands.doctor import _mask as _mask  # noqa: E402
from decoding_sandbox.cli.commands.doctor import (  # noqa: E402
    _report_local_engines as _report_local_engines,
)
from decoding_sandbox.cli.commands.doctor import (  # noqa: E402
    _report_remote_servers as _report_remote_servers,
)
from decoding_sandbox.cli.commands.doctor import cmd_doctor as cmd_doctor  # noqa: E402
from decoding_sandbox.cli.commands.generate import cmd_generate as cmd_generate  # noqa: E402
from decoding_sandbox.cli.commands.inspect import cmd_inspect as cmd_inspect  # noqa: E402
from decoding_sandbox.cli.commands.manual import cmd_manual as cmd_manual  # noqa: E402
from decoding_sandbox.cli.commands.probe import cmd_probe as cmd_probe  # noqa: E402
from decoding_sandbox.cli.commands.serve import cmd_serve as cmd_serve  # noqa: E402
from decoding_sandbox.cli.commands.session import cmd_session as cmd_session  # noqa: E402
from decoding_sandbox.cli.commands.spec import cmd_spec as cmd_spec  # noqa: E402
from decoding_sandbox.cli.commands.web import cmd_web as cmd_web  # noqa: E402

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
