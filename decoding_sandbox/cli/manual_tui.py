"""Interactive manual-decoding TUI built on prompt_toolkit + rich.

Renders the current text and the next-token candidate table, then reads a
command each step. It wraps the backend-agnostic ManualSession, so all logic
(pick/force/undo/save/load) is shared and testable; this file is only I/O.
"""

from __future__ import annotations

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.table import Table

from decoding_sandbox.cli import render
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.manual import ManualSession

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
            render.token_repr(c.text, 18),
            render.fmt_prob(c.prob),
            f"[{render.confidence_style(c.prob)}]{bar}[/]",
        )
    console.print(table)


def run_manual(backend: Backend, prompt: str, top_k: int = 12) -> int:
    console = Console()
    session = ManualSession(backend, prompt, top_k=top_k)
    ps: PromptSession = PromptSession()
    console.print(HELP)
    while True:
        _render(console, session)
        try:
            raw = ps.prompt("decode> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if raw in ("q", "quit", "exit"):
            break
        if raw in ("?", "help"):
            console.print(HELP)
            continue
        if raw == "" or raw.isdigit():
            rank = int(raw) if raw else 0
            try:
                c = session.pick(rank)
                console.print(f"  picked [green]{render.token_repr(c.text)}[/green]")
            except IndexError as exc:
                console.print(f"[red]{exc}[/red]")
            continue
        if raw == "u":
            tid = session.undo()
            console.print("  undone" if tid is not None else "[yellow]nothing to undo[/yellow]")
            continue
        if raw.startswith("f "):
            appended = session.force_text(raw[2:])
            console.print(
                "  forced " + " ".join(render.token_repr(a.text) for a in appended)
            )
            continue
        if raw.startswith("k "):
            try:
                session.top_k = max(1, int(raw[2:]))
            except ValueError:
                console.print("[red]usage: k <n>[/red]")
            continue
        if raw.startswith("s "):
            session.save(raw[2:].strip())
            console.print(f"  saved -> {raw[2:].strip()}")
            continue
        if raw.startswith("l "):
            session.load(raw[2:].strip())
            console.print(f"  loaded <- {raw[2:].strip()}")
            continue
        console.print("[yellow]unknown command (type ? for help)[/yellow]")

    console.print(f"\n[bold]final:[/bold] {session.prompt}[green]{session.generated_text()}[/green]")
    backend.close()
    return 0
