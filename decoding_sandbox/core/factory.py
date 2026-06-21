"""Construct a Backend from a name + Config.

Local (in-process) backends:
- ``hf``          : HF transformers (full vocab, PyTorch); the white-box
                    engine for models that load on the local GPU.
- ``llamacpp``    : HTTP client to a running ``llama-server`` (top-k only).
- ``llamacpp-py`` : In-process ``llama-cpp-python`` (full vocab via
                    ``logits_all=True``) -- the white-box engine for GGUFs
                    that HF can't host (e.g. Qwen3.5-9B on the 6 GB Pascal).

Remote backends:
- ``remote``      : HTTP client to a ``dsbx serve`` instance. Reads
                    ``[remote.<run.backend>]`` or the first
                    ``[remote.NAME]`` entry if ``remote`` is asked for
                    generically.
- ``<NAME>``      : Any name matching a configured ``[remote.NAME]`` block
                    routes to ``RemoteBackend`` against that entry's
                    ``base_url``. This is the recommended form (e.g.
                    ``dsbx-host-py``, ``dsbx-host-hf``) so config and CLI line up.

Cloud backends (``fireworks``/``nim``/``openrouter``/``lmstudio``) route
through OpenAICompatBackend.
"""

from __future__ import annotations

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config

LOCAL_BACKENDS = ("hf", "llamacpp", "llamacpp-py")


def _normalize(name: str) -> str:
    # Accept the dash-form ``llamacpp-py`` and the TOML/identifier-friendly
    # ``llamacpp_py``; everything else is case-folded.
    return name.lower().replace("_", "-")


def build_backend(name: str, cfg: Config, model: str | None = None) -> Backend:
    norm = _normalize(name)
    if norm == "hf":
        from decoding_sandbox.backends.hf import HFBackend

        hf = cfg.get("local", "hf", default={})
        return HFBackend(
            model or hf.get("model", "Qwen/Qwen3-1.7B-Base"),
            fallback_model=hf.get("fallback_model"),
            load_in_4bit=bool(hf.get("load_in_4bit", True)),
            gpu_mem=str(hf.get("gpu_mem", "4500MiB")),
            cpu_mem=str(hf.get("cpu_mem", "13GiB")),
        )
    if norm in ("llamacpp", "llama"):
        from decoding_sandbox.backends.llamacpp import LlamaCppBackend

        lc = cfg.get("local", "llamacpp", default={})
        return LlamaCppBackend(lc.get("base_url", "http://127.0.0.1:8080"))

    if norm in ("llamacpp-py", "llamacpp-python", "llama-py"):
        from decoding_sandbox.backends.llamacpp_py import LlamaCppPyBackend

        lp = cfg.get("local", "llamacpp_py", default={})
        return LlamaCppPyBackend(
            model_path=model or lp.get("model_path"),
            model_glob=lp.get("model_glob", "**/*.gguf"),
            model_search_dirs=list(lp.get("model_search_dirs", [])),
            n_gpu_layers=int(lp.get("n_gpu_layers", 20)),
            n_ctx=int(lp.get("n_ctx", 4096)),
            logits_all=bool(lp.get("logits_all", True)),
            verbose=bool(lp.get("verbose", False)),
        )

    # Remote / provider lookups follow.
    # Lookups are case-insensitive: we try the normalized form first; if
    # that doesn't match (e.g. provider names legitimately contain
    # underscores), fall back to the lowered original.
    for candidate in (norm, name.lower()):
        if candidate in cfg.remotes:
            return _build_remote(cfg.remotes[candidate])

    if norm == "remote":
        # Generic ``--backend remote`` picks the single configured entry.
        # Ambiguous when several exist; the user should name one
        # explicitly (e.g. ``--backend dsbx-host-py``).
        if not cfg.remotes:
            raise ValueError(
                "Backend 'remote' requested but no [remote.NAME] blocks "
                "are configured. Add one to config.toml, e.g.\n\n"
                "  [remote.dsbx-host-py]\n"
                '  base_url = "http://192.0.2.42:8000"\n'
            )
        if len(cfg.remotes) > 1:
            raise ValueError(
                f"Backend 'remote' is ambiguous: multiple [remote.NAME] "
                f"blocks configured ({sorted(cfg.remotes)}). Pass "
                "``--backend <NAME>`` to pick one."
            )
        only = next(iter(cfg.remotes.values()))
        return _build_remote(only)

    for candidate in (norm, name.lower()):
        if candidate in cfg.providers:
            from decoding_sandbox.backends.openai_compat import OpenAICompatBackend

            return OpenAICompatBackend(cfg.provider(candidate), model=model)

    available = list(LOCAL_BACKENDS) + sorted(cfg.remotes) + sorted(cfg.providers)
    raise ValueError(f"Backend '{name}' not available. Choose from: {available}")


def _build_remote(rc) -> Backend:
    """Build a ``RemoteBackend`` from a :class:`RemoteConfig` entry.

    Kept in a helper so the factory has a single import point for the
    optional dependency (httpx is core; the server itself is the optional
    part) and the error message stays uniform regardless of which name
    triggered the lookup.
    """
    from decoding_sandbox.backends.remote import RemoteBackend

    return RemoteBackend(rc.base_url, timeout=rc.timeout)
