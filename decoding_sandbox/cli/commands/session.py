"""``dsbx session`` -- long-lived REPL that keeps the model loaded."""

from __future__ import annotations

import argparse
import contextlib

from decoding_sandbox.cli import app
from decoding_sandbox.cli._shared import (
    _build_backend_with_load_timing,
    _print_backend_banner,
)
from decoding_sandbox.cli.timing import Timing
from decoding_sandbox.core.config import Config


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

    rc = app._run_preflight(cfg, skip=getattr(args, "skip_preflight", False))
    if rc is not None:
        return rc

    name = args.backend or cfg.default_backend
    timing = None if getattr(args, "no_timing", False) else Timing()
    backend = _build_backend_with_load_timing(name, cfg, model=args.model, timing=timing)
    _print_backend_banner(backend)
    if timing is not None:
        app.console.print(timing.render(prefix="startup"))

    state = SessionState(
        cfg=cfg,
        backend=backend,
        backend_name=name,
        backend_model=args.model,
        console=app.console,
        timing_enabled=not getattr(args, "no_timing", False),
    )
    try:
        return run_session(state)
    finally:
        with contextlib.suppress(Exception):
            if state.backend is not None:
                state.backend.close()
