"""``dsbx probe`` -- live provider logprob capability check."""

from __future__ import annotations

import argparse

from decoding_sandbox.cli import app
from decoding_sandbox.core.config import Config


def cmd_probe(args: argparse.Namespace, cfg: Config) -> int:
    from decoding_sandbox.core import provider_probe

    return provider_probe.run_probe(
        cfg,
        providers=args.providers,
        model=args.model,
        console=app.console,
    )
