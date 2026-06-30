"""``python -m dsbx.server`` launcher.

The public entry point for running the server is ``dsbx serve`` (see
:mod:`dsbx.cli.app`). This module exists so the same code path
is reachable without the CLI -- useful for ad-hoc testing and for systemd
unit files that prefer ``ExecStart=python -m dsbx.server ...``.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dsbx.server",
        description="Run the dsbx HTTP server (FastAPI + uvicorn).",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("hf", "llamacpp-py"),
        help="Which heavy in-process backend to host.",
    )
    parser.add_argument("--model", default=None, help="Override the model id / GGUF path.")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help=(
            "Start with no model loaded (empty slot). Use POST /v1/reload "
            "(or the web UI) to load one on demand."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None, help="Path to a config.toml override.")
    args = parser.parse_args(argv)

    # Heavy imports are deferred so --help is snappy and importing this
    # module never pulls fastapi/uvicorn into the test environment.
    import uvicorn

    from dsbx.core.config import load_config
    from dsbx.core.factory import build_backend, list_available_models
    from dsbx.server import schemas as S
    from dsbx.server.app import make_app

    cfg = load_config(args.config)

    def builder(model: str | None):
        return build_backend(args.backend, cfg, model=model)

    def model_lister() -> list[S.ServerModelEntry]:
        from pathlib import Path as _Path

        def _size(model_id: str) -> int | None:
            # GGUF ids are real file paths; expose their on-disk size so the
            # UI can draw a size-proportional load bar. HF repo ids aren't
            # local files -> size stays None.
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

    if args.no_preload:
        app = make_app(
            backend_kind=args.backend,
            builder=builder,
            model=args.model,
            model_lister=model_lister,
            preload=False,
        )
    else:
        backend = builder(args.model)
        app = make_app(
            backend,
            backend_kind=args.backend,
            builder=builder,
            model=args.model,
            model_lister=model_lister,
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
