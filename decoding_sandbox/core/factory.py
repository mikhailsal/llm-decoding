"""Construct a Backend from a name + Config.

Local backends:
- ``hf``          : HF transformers (full vocab, PyTorch); the white-box
                    engine for models that load on the local GPU.
- ``llamacpp``    : HTTP client to a running ``llama-server`` (top-k only).
- ``llamacpp-py`` : In-process ``llama-cpp-python`` (full vocab via
                    ``logits_all=True``) -- the white-box engine for GGUFs
                    that HF can't host (e.g. Qwen3.5-9B on the 6 GB Pascal).

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

    # Providers are looked up case-insensitively too. We try the normalized
    # form first; if that doesn't match a provider (e.g. provider names
    # legitimately contain underscores), fall back to the lowered original.
    for candidate in (norm, name.lower()):
        if candidate in cfg.providers:
            from decoding_sandbox.backends.openai_compat import OpenAICompatBackend

            return OpenAICompatBackend(cfg.provider(candidate), model=model)

    available = list(LOCAL_BACKENDS) + sorted(cfg.providers)
    raise ValueError(f"Backend '{name}' not available. Choose from: {available}")
