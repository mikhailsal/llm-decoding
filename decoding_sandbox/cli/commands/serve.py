"""``dsbx serve`` -- launch the dsbx HTTP server wrapping one backend."""

from __future__ import annotations

import argparse
import contextlib

from decoding_sandbox.cli import app
from decoding_sandbox.cli._shared import _build_backend_with_load_timing
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config


def cmd_serve(args: argparse.Namespace, cfg: Config) -> int:
    """Launch the dsbx HTTP server (FastAPI + uvicorn) wrapping one backend.

    Heavy imports (``fastapi``/``uvicorn``) live inside the function so
    they remain optional: the rest of the CLI keeps working on machines
    that only have the core ``[project.dependencies]`` installed.

    The server hosts a single in-process backend for its lifetime. Pair
    one ``dsbx serve --backend llamacpp-py`` and one
    ``dsbx serve --backend hf`` on different ports if you want both
    available simultaneously -- the client picks via ``[remote.NAME]``
    aliases in ``config.toml``.
    """
    try:
        import uvicorn  # type: ignore
    except ImportError as exc:
        app.console.print(
            "[red]dsbx serve requires the [bold]server[/bold] extra. "
            'Install with: [cyan]pip install -e ".[server]"[/cyan][/red]'
        )
        app.console.print(f"[dim]underlying error: {exc}[/dim]")
        return 2

    # Loopback is the only safe default: there's no auth, anyone on the
    # host network can talk to a loaded model. We *allow* opting in to
    # public binding with --host 0.0.0.0 (typical for the client <->
    # dsbx-host LAN case), but we make it visible so it's never accidental.
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        app.console.print(
            f"[yellow]warning:[/yellow] binding to [bold]{args.host}[/bold] "
            "(not loopback). The server has no authentication; anyone who "
            "can reach this address can drive the loaded model."
        )

    from decoding_sandbox.core.factory import build_backend, list_available_models
    from decoding_sandbox.server import schemas as S
    from decoding_sandbox.server.app import make_app

    # Builder + catalogue closures captured for the swappable model slot.
    # The builder reuses the exact factory path so a browser-driven reload
    # constructs the new model identically to a fresh ``dsbx serve``.
    def builder(model: str | None) -> Backend:
        return build_backend(args.backend, cfg, model=model)

    def model_lister() -> list[S.ServerModelEntry]:
        from pathlib import Path as _Path

        def _size(model_id: str) -> int | None:
            # GGUF ids are real file paths -> expose on-disk size for the
            # UI's size-proportional load bar; HF repo ids stay None.
            p = _Path(model_id)
            try:
                if p.is_file():
                    return p.stat().st_size
            except OSError:
                return None
            return None

        return [
            S.ServerModelEntry(id=i, label=label, size_bytes=_size(i))
            for i, label in list_available_models(args.backend, cfg)
        ]

    if getattr(args, "no_preload", False):
        app.console.print(
            f"[dim]starting '{args.backend}' server with no model preloaded "
            "(--no-preload); load one from the web UI.[/dim]"
        )
        server_app = make_app(
            backend_kind=args.backend,
            builder=builder,
            model=args.model,
            model_lister=model_lister,
            preload=False,
        )
        app.console.print(
            f"  serving on [bold]http://{args.host}:{args.port}[/bold] [dim](empty slot)[/dim]"
        )
        uvicorn.run(server_app, host=args.host, port=args.port, log_level=args.log_level)
        return 0

    app.console.print(f"[dim]building backend '{args.backend}' for the server...[/dim]")
    backend = _build_backend_with_load_timing(args.backend, cfg, model=args.model, timing=None)
    app.console.print(
        f"  loaded [cyan]{backend.capabilities.name}[/cyan] -- "
        f"serving on [bold]http://{args.host}:{args.port}[/bold]"
    )
    server_app = make_app(
        backend,
        backend_kind=args.backend,
        builder=builder,
        model=args.model,
        model_lister=model_lister,
    )
    try:
        uvicorn.run(server_app, host=args.host, port=args.port, log_level=args.log_level)
    finally:
        with contextlib.suppress(Exception):
            backend.close()
    return 0
