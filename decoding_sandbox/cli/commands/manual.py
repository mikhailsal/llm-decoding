"""``dsbx manual`` -- interactive token-by-token TUI (prompt_toolkit)."""

from __future__ import annotations

import argparse

from decoding_sandbox.cli import app
from decoding_sandbox.cli._shared import _build_backend_with_load_timing
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config


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
        rc = app._run_preflight(cfg, skip=getattr(args, "skip_preflight", False))
        if rc is not None:
            return rc
        name = args.backend or cfg.default_backend
        backend = _build_backend_with_load_timing(name, cfg, model=args.model, timing=None)
    assert backend is not None  # set above when ``own_backend`` was True
    return run_manual(backend, args.prompt, top_k=args.top_k, own_backend=own_backend)
