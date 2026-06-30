"""``dsbx probe`` -- live provider logprob capability check."""

from __future__ import annotations

import argparse

from dsbx.cli import app
from dsbx.core.config import Config


def cmd_probe(args: argparse.Namespace, cfg: Config) -> int:
    from dsbx.core import provider_probe

    return provider_probe.run_probe(
        cfg,
        providers=args.providers,
        model=args.model,
        console=app.console,
    )
