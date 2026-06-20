"""Construct a Backend from a name + Config.

Local backends (hf, llamacpp) are implemented in Wave 1. Cloud backends
(fireworks/nim/openrouter via OpenAICompatBackend) arrive in Wave 4.
"""

from __future__ import annotations

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config

LOCAL_BACKENDS = ("hf", "llamacpp")


def build_backend(name: str, cfg: Config, model: str | None = None) -> Backend:
    name = name.lower()
    if name == "hf":
        from decoding_sandbox.backends.hf import HFBackend

        hf = cfg.get("local", "hf", default={})
        return HFBackend(
            model or hf.get("model", "Qwen/Qwen3-1.7B-Base"),
            fallback_model=hf.get("fallback_model"),
            load_in_4bit=bool(hf.get("load_in_4bit", True)),
            gpu_mem=str(hf.get("gpu_mem", "4500MiB")),
            cpu_mem=str(hf.get("cpu_mem", "13GiB")),
        )
    if name in ("llamacpp", "llama"):
        from decoding_sandbox.backends.llamacpp import LlamaCppBackend

        lc = cfg.get("local", "llamacpp", default={})
        return LlamaCppBackend(lc.get("base_url", "http://127.0.0.1:8080"))

    if name in cfg.providers:
        from decoding_sandbox.backends.openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(cfg.provider(name), model=model)

    available = list(LOCAL_BACKENDS) + sorted(cfg.providers)
    raise ValueError(f"Backend '{name}' not available. Choose from: {available}")
