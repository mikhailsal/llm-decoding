"""``dsbx web`` -- launch the dsbx web middleware (FastAPI + uvicorn)."""

from __future__ import annotations

import argparse
import os

from dsbx import __version__
from dsbx.cli import app
from dsbx.core.config import Config


def cmd_web(args: argparse.Namespace, cfg: Config) -> int:
    """Launch the dsbx web middleware (FastAPI + uvicorn).

    The middleware fronts every configured backend behind a single
    bearer-token API so the browser never sees provider keys or remote
    server URLs. Token resolution order: ``--token`` > ``$DSBX_WEB_TOKEN``
    > ``[web].api_token`` in config.toml.

    Heavy imports live inside the function so the rest of the CLI keeps
    working on machines that only have the core dependencies installed.
    """
    try:
        import uvicorn  # type: ignore
    except ImportError as exc:
        app.console.print(
            "[red]dsbx web requires the [bold]web[/bold] extra. "
            'Install with: [cyan]pip install -e ".[web]"[/cyan][/red]'
        )
        app.console.print(f"[dim]underlying error: {exc}[/dim]")
        return 2

    from dsbx.web.app import make_web_app

    web_cfg = cfg.get("web", default={}) or {}
    token = (
        args.token
        or os.environ.get("DSBX_WEB_TOKEN")
        or str(web_cfg.get("api_token") or "").strip()
    )
    if not token:
        app.console.print(
            "[red]dsbx web requires a bearer token.[/red] Set one via "
            "[cyan]--token[/cyan], [cyan]$DSBX_WEB_TOKEN[/cyan], or "
            "[cyan][web].api_token[/cyan] in config.toml. A long random "
            "string is best (e.g. [dim]openssl rand -hex 32[/dim])."
        )
        return 2

    cors_origins = list(web_cfg.get("cors_origins") or [])
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        app.console.print(
            f"[yellow]warning:[/yellow] binding to [bold]{args.host}[/bold] "
            "(not loopback). The middleware authenticates requests, but make "
            "sure the bearer token is strong and the box isn't exposed to "
            "the public internet."
        )

    manual_ttl = float(web_cfg.get("manual_session_ttl", 3600.0))
    web_app = make_web_app(
        cfg,
        token=token,
        server_label=args.server_label,
        cors_origins=cors_origins,
        frontend_dist=args.frontend_dist,
        manual_ttl_seconds=manual_ttl,
    )

    app.console.print(
        f"[dim]dsbx web {__version__} -- serving on [bold]"
        f"http://{args.host}:{args.port}[/bold][/dim]"
    )
    app.console.print(
        f"[dim]bearer token: {token[:4]}...{token[-3:] if len(token) > 8 else ''}"
        f" ({len(token)} chars)[/dim]"
    )
    if args.frontend_dist:
        app.console.print(f"[dim]frontend bundle: [cyan]{args.frontend_dist}[/cyan][/dim]")
    uvicorn.run(web_app, host=args.host, port=args.port, log_level=args.log_level)
    return 0
