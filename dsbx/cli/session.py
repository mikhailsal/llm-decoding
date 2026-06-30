"""Long-lived REPL that keeps a backend loaded across commands.

``dsbx session`` builds a backend once and then accepts a small grammar of
follow-up commands -- ``inspect``, ``generate``, ``manual``, ``spec`` (which
loads its own pair), plus meta-commands like ``:caps``, ``:backend NAME``,
``:timing on|off``, ``:help``, ``:quit``. Every command reuses the same
backend instance, so the 30+ seconds spent loading a 9B GGUF amortizes
across every subsequent inspect/generate.

The command grammar mirrors the non-interactive subparsers so users can
copy-paste examples from ``dsbx inspect --help`` and have them work
verbatim. We do that by reusing the same subparsers from ``build_parser``
and just dispatching the parsed args through our session-aware versions of
``cmd_inspect``/``cmd_generate``/``cmd_manual``/``cmd_spec``.

The pure dispatch entry point (``dispatch_session_line``) is decoupled from
the prompt-toolkit loop so it can be unit-tested deterministically.
"""

from __future__ import annotations

import argparse
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from dsbx.core.backend import Backend
    from dsbx.core.config import Config


@dataclass
class SessionState:
    """Mutable state held by the REPL loop.

    The backend can be swapped via ``:backend NAME``: we close the old one
    and build a new one (using the same Config the session was started
    with). ``timing_enabled`` mirrors what individual commands' ``--no-timing``
    flag would do; flip it inline with ``:timing on|off``.
    """

    cfg: Config
    backend: Backend | None
    backend_name: str
    backend_model: str | None
    console: Console
    timing_enabled: bool = True
    history: list[str] = field(default_factory=list)


@dataclass
class DispatchResult:
    tag: str  # "ok" | "quit" | "unknown" | "parse_error" | "meta"
    exit_code: int = 0
    message: str = ""

    @property
    def should_quit(self) -> bool:
        return self.tag == "quit"


# --------------------------------------------------------------------------- #
# Argparse for the in-session command grammar.
# --------------------------------------------------------------------------- #
class _NoSysExitParser(argparse.ArgumentParser):
    """Raise instead of calling ``sys.exit`` -- the REPL must keep running."""

    def error(self, message):
        raise argparse.ArgumentError(None, message)


def _build_session_parser() -> argparse.ArgumentParser:
    """The parser used inside the REPL. Mirrors ``app.build_parser``'s
    inspect/generate/manual/spec but without the global ``--config`` arg
    (the session was launched with one and we keep it for every command).
    """
    # We import lazily to avoid an import cycle on module load.
    from dsbx.cli.app import (
        _BACKEND_HELP,
        _add_preflight_flag,
        cmd_generate,
        cmd_inspect,
        cmd_manual,
        cmd_spec,
    )

    parser = _NoSysExitParser(
        prog="dsbx",
        description="(session) commands",
        add_help=False,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", add_help=False)
    p_inspect.add_argument("prompt")
    p_inspect.add_argument("--backend", default=None, help=_BACKEND_HELP)
    p_inspect.add_argument("--model", default=None)
    p_inspect.add_argument("--top-k", type=int, default=8)
    p_inspect.add_argument("--watch", action="append", default=[])
    p_inspect.add_argument("--watch-id", action="append", type=int, default=[])
    p_inspect.add_argument("--watch-eos", action="store_true", default=False)
    p_inspect.add_argument("--candidates", type=int, default=0)
    p_inspect.add_argument("--no-timing", action="store_true")
    _add_preflight_flag(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect, skip_preflight=True)  # session already did it

    p_gen = sub.add_parser("generate", add_help=False)
    p_gen.add_argument("prompt")
    p_gen.add_argument("--backend", default=None, help=_BACKEND_HELP)
    p_gen.add_argument("--model", default=None)
    p_gen.add_argument(
        "--sampler",
        default="greedy",
        choices=["greedy", "temperature", "top_k", "top_p", "min_p", "typical", "custom"],
    )
    p_gen.add_argument("--custom-file", default=None)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--sampler-top-k", type=int, default=None)
    p_gen.add_argument("--top-p", type=float, default=None)
    p_gen.add_argument("--min-p", type=float, default=None)
    p_gen.add_argument("--typical-p", type=float, default=None)
    p_gen.add_argument("--max-tokens", type=int, default=20)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--top-k", type=int, default=50)
    p_gen.add_argument("--stop", action="append", default=[])
    p_gen.add_argument("--no-timing", action="store_true")
    _add_preflight_flag(p_gen)
    p_gen.set_defaults(func=cmd_generate, skip_preflight=True)

    p_manual = sub.add_parser("manual", add_help=False)
    p_manual.add_argument("prompt")
    p_manual.add_argument("--backend", default=None, help=_BACKEND_HELP)
    p_manual.add_argument("--model", default=None)
    p_manual.add_argument("--top-k", type=int, default=12)
    _add_preflight_flag(p_manual)
    p_manual.set_defaults(func=cmd_manual, skip_preflight=True)

    p_spec = sub.add_parser("spec", add_help=False)
    p_spec.add_argument("prompt")
    p_spec.add_argument("--target-model", default="Qwen/Qwen3-1.7B-Base")
    p_spec.add_argument("--draft-model", default="Qwen/Qwen3-0.6B-Base")
    p_spec.add_argument("--gamma", type=int, default=4)
    p_spec.add_argument("--max-tokens", type=int, default=24)
    p_spec.add_argument("--no-timing", action="store_true")
    _add_preflight_flag(p_spec)
    p_spec.set_defaults(func=cmd_spec, skip_preflight=True)

    return parser


HELP_TEXT = """\
session commands:
  inspect "<text>" [--watch ' Paris'] [--watch-id N] [--watch-eos]
                   [--top-k N] [--candidates N]
  generate "<text>" [--sampler greedy|temperature|top_k|top_p|min_p|typical|custom]
                    [--temperature T] [--top-p P] [--min-p P] [--typical-p P]
                    [--sampler-top-k K] [--max-tokens N] [--seed N]
                    [--top-k K] [--stop ' END']
  manual "<text>" [--top-k N]
  spec    "<text>" [--target-model M] [--draft-model M] [--gamma G] [--max-tokens N]

meta:
  :caps              show the current backend's capability banner
  :backend NAME [MODEL]
                     switch the loaded backend (closes the old one)
  :timing on|off     toggle the timing summary printed after each command
  :history           show this session's command history
  :help / :?         show this help
  :quit / :exit      leave the session
"""


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def dispatch_session_line(
    state: SessionState,
    raw: str,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> DispatchResult:
    """Apply a single REPL line to ``state``.

    The pure entry point: takes the user's typed line, mutates ``state``
    (e.g. on ``:backend``), and returns a ``DispatchResult`` so tests can
    assert on what happened without a real terminal.
    """
    parser = parser or _build_session_parser()
    line = raw.strip()
    if not line:
        return DispatchResult("ok")
    state.history.append(line)

    if line in (":quit", ":exit", ":q"):
        return DispatchResult("quit", message="bye")
    if line in (":help", ":?"):
        state.console.print(HELP_TEXT)
        return DispatchResult("meta")
    if line == ":caps":
        from dsbx.cli.app import _print_backend_banner

        _print_backend_banner(state.backend, out=state.console)  # type: ignore[arg-type]
        return DispatchResult("meta")
    if line == ":history":
        for i, h in enumerate(state.history[:-1], 1):
            state.console.print(f"  {i:3d}  {h}")
        return DispatchResult("meta")
    if line.startswith(":timing"):
        rest = line[len(":timing") :].strip().lower()
        if rest in ("on", "1", "true", "yes"):
            state.timing_enabled = True
        elif rest in ("off", "0", "false", "no"):
            state.timing_enabled = False
        elif rest == "":
            pass  # just report
        else:
            return DispatchResult(
                "parse_error",
                exit_code=2,
                message=f"unknown :timing argument {rest!r} (use on/off)",
            )
        state.console.print(f"  timing: [green]{'on' if state.timing_enabled else 'off'}[/green]")
        return DispatchResult("meta")
    if line.startswith(":backend"):
        return _switch_backend(state, line[len(":backend") :])

    # Regular subcommand. Use shlex so quoted prompts come through intact.
    try:
        argv = shlex.split(line)
    except ValueError as exc:
        return DispatchResult("parse_error", exit_code=2, message=f"shell-parse error: {exc}")
    try:
        args = parser.parse_args(argv)
    except argparse.ArgumentError as exc:
        return DispatchResult("parse_error", exit_code=2, message=str(exc))

    # Mirror the standalone CLI's --no-timing semantics through to the
    # underlying handlers via the same args namespace.
    if not getattr(args, "no_timing", False):
        args.no_timing = not state.timing_enabled

    rc = args.func(args, state.cfg, backend=state.backend, show_banner=False)
    return DispatchResult("ok", exit_code=int(rc))


def _switch_backend(state: SessionState, rest: str) -> DispatchResult:
    """Handle ``:backend NAME [MODEL]``. Closes the old backend on success."""
    parts = rest.split()
    if not parts:
        state.console.print(
            f"  current backend: [cyan]{state.backend_name}[/cyan]"
            + (f" (model={state.backend_model})" if state.backend_model else "")
        )
        return DispatchResult("meta")
    name = parts[0]
    model = " ".join(parts[1:]) if len(parts) > 1 else None

    # Close the old one first so its VRAM is released before we load.
    old = state.backend
    state.console.print("[dim]closing previous backend...[/dim]")
    try:
        if old is not None:
            old.close()
    except Exception as exc:
        state.console.print(f"[yellow]close warning: {exc}[/yellow]")

    state.console.print(f"[dim]building backend '{name}'...[/dim]")
    t0 = time.perf_counter()
    try:
        from dsbx.core.factory import build_backend

        state.backend = build_backend(name, state.cfg, model=model)  # type: ignore[arg-type]
    except Exception as exc:
        return DispatchResult("parse_error", exit_code=1, message=f"failed to build backend: {exc}")
    state.backend_name = name
    state.backend_model = model
    state.console.print(
        f"  switched to [cyan]{state.backend.capabilities.name}[/cyan]  "  # type: ignore[union-attr]
        f"(load {time.perf_counter() - t0:.1f} s)"
    )
    return DispatchResult("meta")


# --------------------------------------------------------------------------- #
# Interactive loop (the actual REPL, separate from dispatch for testability).
# --------------------------------------------------------------------------- #
def run_session(state: SessionState, *, prompt_func: Callable[[str], str] | None = None) -> int:
    """REPL loop. ``prompt_func`` is injectable so tests can drive the loop
    without a real terminal."""
    if prompt_func is None:
        from prompt_toolkit import PromptSession

        ps: PromptSession = PromptSession()
        prompt_func = ps.prompt  # type: ignore[assignment]

    state.console.print("[dim]session ready. type ':help' for commands, ':quit' to exit.[/dim]")
    while True:
        try:
            line = prompt_func("dsbx> ")
        except (EOFError, KeyboardInterrupt):
            state.console.print()
            break
        result = dispatch_session_line(state, line)
        if result.message and result.tag in ("parse_error", "unknown"):
            state.console.print(f"[red]{result.message}[/red]")
        if result.should_quit:
            break
    return 0
