"""Interactive manual-decoding TUI built on prompt_toolkit + rich.

Renders the current text and the next-token candidate table, then reads a
command each step. It wraps the backend-agnostic ManualSession, so all logic
(pick/force/undo/save/load) is shared and testable; this file is only I/O.

The command dispatcher is factored out as ``dispatch_command`` so it can be
unit-tested without a real terminal: it returns a small tagged dataclass
(``CommandResult``) describing what happened, which the TUI converts to rich
output. New commands go in ``dispatch_command`` and stay testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.table import Table

from dsbx.cli import render
from dsbx.core.backend import Backend
from dsbx.core.manual import ManualSession

HELP = """\
commands:
  <number>     pick the candidate with that rank (Enter = rank 0 / greedy)
  f <text>     force an arbitrary string (e.g. f  Berlin) -- even an unlikely token
  u            undo the last token
  k <n>        set how many candidates to show
  s <path>     save transcript to JSON
  l <path>     load transcript from JSON
  ?            show this help
  q            quit
"""

# Tagged outcome of a single command, used by both the live TUI and tests.
Tag = Literal[
    "quit",
    "help",
    "pick",
    "pick_error",
    "undo",
    "undo_empty",
    "force",
    "force_blocked",
    "set_top_k",
    "bad_top_k",
    "save",
    "load",
    "unknown",
]


@dataclass
class CommandResult:
    tag: Tag
    message: str = ""

    @property
    def should_quit(self) -> bool:
        return self.tag == "quit"


def dispatch_command(session: ManualSession, raw: str) -> CommandResult:
    """Apply one user command to ``session`` and return what happened.

    Pure I/O-free entry point so unit tests can exercise the full grammar
    without standing up a PromptSession or a rich Console.
    """
    raw = raw.strip()

    if raw in ("q", "quit", "exit"):
        return CommandResult("quit")
    if raw in ("?", "help"):
        return CommandResult("help", HELP)

    if raw == "" or raw.isdigit():
        rank = int(raw) if raw else 0
        try:
            c = session.pick(rank)
        except IndexError as exc:
            return CommandResult("pick_error", str(exc))
        return CommandResult("pick", c.text)

    if raw == "u":
        tid = session.undo()
        if tid is None:
            return CommandResult("undo_empty")
        return CommandResult("undo", str(tid))

    if raw.startswith("f "):
        if not session.backend.capabilities.can_force_token:
            return CommandResult(
                "force_blocked",
                f"backend {session.backend.capabilities.name!r} cannot force "
                "arbitrary tokens (capabilities.can_force_token=False).",
            )
        appended = session.force_text(raw[2:])
        return CommandResult("force", " ".join(a.text for a in appended))

    if raw.startswith("k "):
        try:
            session.top_k = max(1, int(raw[2:]))
        except ValueError:
            return CommandResult("bad_top_k", "usage: k <n>")
        return CommandResult("set_top_k", str(session.top_k))

    if raw.startswith("s "):
        path = raw[2:].strip()
        session.save(path)
        return CommandResult("save", path)

    if raw.startswith("l "):
        path = raw[2:].strip()
        session.load(path)
        return CommandResult("load", path)

    return CommandResult("unknown")


def _render(console: Console, session: ManualSession) -> None:
    console.rule("[bold]manual decoding[/bold]")
    gen = session.generated_text()
    console.print(
        f"[dim]prompt:[/dim] {session.prompt}"
        + (f"[green]{render.token_repr(gen)}[/green]" if gen else "")
    )
    dist = session.distribution()
    table = Table(title=f"next-token candidates (top {session.top_k})")
    table.add_column("rank", justify="right")
    table.add_column("token")
    table.add_column("prob", justify="right")
    table.add_column("bar")
    for c in dist.candidates:
        bar = "█" * max(1, int(c.prob * 30)) if c.prob == c.prob else ""
        table.add_row(
            str(c.rank),
            render.token_repr(c.text, 18, is_special=c.is_special),
            render.fmt_prob(c.prob),
            f"[{render.confidence_style(c.prob)}]{bar}[/]",
        )
    console.print(table)


def _print_result(console: Console, result: CommandResult) -> None:
    tag = result.tag
    if tag == "help":
        console.print(result.message)
    elif tag == "pick":
        console.print(f"  picked [green]{render.token_repr(result.message)}[/green]")
    elif tag == "pick_error":
        console.print(f"[red]{result.message}[/red]")
    elif tag == "undo":
        console.print("  undone")
    elif tag == "undo_empty":
        console.print("[yellow]nothing to undo[/yellow]")
    elif tag == "force":
        console.print(f"  forced {render.token_repr(result.message)}")
    elif tag == "force_blocked":
        console.print(f"[yellow]{result.message} pick by rank instead.[/yellow]")
    elif tag == "set_top_k":
        console.print(f"  top_k={result.message}")
    elif tag == "bad_top_k":
        console.print(f"[red]{result.message}[/red]")
    elif tag == "save":
        console.print(f"  saved -> {result.message}")
    elif tag == "load":
        console.print(f"  loaded <- {result.message}")
    elif tag == "unknown":
        console.print("[yellow]unknown command (type ? for help)[/yellow]")


def run_manual(backend: Backend, prompt: str, top_k: int = 12, *, own_backend: bool = True) -> int:
    """Run the interactive manual-decoding TUI.

    When ``own_backend=False`` (e.g. called from the long-lived ``session``
    REPL), the backend is left open on return so the parent can keep using
    it. Otherwise the backend is closed at the end as before.

    The rich Console is borrowed from ``app.console`` so the user's
    ``--color always/never/auto`` choice (set in ``main``) flows through
    -- otherwise the manual TUI would always default to rich's TTY
    auto-detection regardless of the CLI flag.
    """
    from dsbx.cli.app import console as _app_console

    console = _app_console
    session = ManualSession(backend, prompt, top_k=top_k)
    ps: PromptSession = PromptSession()
    console.print(HELP)
    while True:
        _render(console, session)
        try:
            raw = ps.prompt("decode> ")
        except (EOFError, KeyboardInterrupt):
            break

        result = dispatch_command(session, raw)
        _print_result(console, result)
        if result.should_quit:
            break

    console.print(
        f"\n[bold]final:[/bold] {session.prompt}[green]{session.generated_text()}[/green]"
    )
    if own_backend:
        backend.close()
    return 0
