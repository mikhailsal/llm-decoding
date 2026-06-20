"""Tests for the backend factory.

We don't actually load HF/torch (the dsbx-host-only stack), but we do verify the
factory routes names correctly, threads config through to constructors, and
exposes provider backends via OpenAICompatBackend.
"""

from __future__ import annotations

import pytest

from decoding_sandbox.core import factory as factory_mod
from decoding_sandbox.core.config import load_config


def test_build_backend_unknown_name_raises_value_error() -> None:
    cfg = load_config(load_secrets=False)
    with pytest.raises(ValueError, match="Backend 'nope' not available"):
        factory_mod.build_backend("nope", cfg)


def test_build_backend_lowercases_name() -> None:
    """Provider names look up case-insensitively."""
    cfg = load_config(load_secrets=False)
    # We can't actually instantiate Fireworks without httpx hitting the wire,
    # so monkey-test the wiring by stubbing the OpenAICompatBackend constructor.
    import decoding_sandbox.backends.openai_compat as oc

    called: dict = {}

    class _Stub:
        def __init__(self, provider, model=None):
            called["provider"] = provider
            called["model"] = model

    orig = oc.OpenAICompatBackend
    oc.OpenAICompatBackend = _Stub
    try:
        factory_mod.build_backend("FIREWORKS", cfg, model="custom-model")
    finally:
        oc.OpenAICompatBackend = orig

    assert called["provider"].name == "fireworks"
    assert called["model"] == "custom-model"


def test_build_backend_llamacpp_alias() -> None:
    cfg = load_config(load_secrets=False)
    import decoding_sandbox.backends.llamacpp as lc

    constructed: dict = {}

    class _Stub:
        def __init__(self, base_url):
            constructed["base_url"] = base_url

    orig = lc.LlamaCppBackend
    lc.LlamaCppBackend = _Stub
    try:
        factory_mod.build_backend("llama", cfg)
        assert constructed["base_url"] == cfg.get("local", "llamacpp", "base_url")
        # Both names route to the same constructor.
        factory_mod.build_backend("llamacpp", cfg)
        assert constructed["base_url"] == cfg.get("local", "llamacpp", "base_url")
    finally:
        lc.LlamaCppBackend = orig


def test_build_backend_hf_forwards_memory_caps() -> None:
    cfg = load_config(load_secrets=False)
    # Make sure the config carries the defaults we expect.
    hf = cfg.get("local", "hf")
    assert hf["gpu_mem"] == "4500MiB"
    assert hf["cpu_mem"] == "13GiB"

    import decoding_sandbox.backends.hf as hf_mod

    captured: dict = {}

    class _Stub:
        def __init__(self, model_id, *, fallback_model=None, load_in_4bit=True,
                     gpu_mem="0MiB", cpu_mem="0MiB"):
            captured["model_id"] = model_id
            captured["fallback_model"] = fallback_model
            captured["load_in_4bit"] = load_in_4bit
            captured["gpu_mem"] = gpu_mem
            captured["cpu_mem"] = cpu_mem

    orig = hf_mod.HFBackend
    hf_mod.HFBackend = _Stub
    try:
        factory_mod.build_backend("hf", cfg, model="Q/q")
    finally:
        hf_mod.HFBackend = orig

    assert captured["model_id"] == "Q/q"
    assert captured["fallback_model"] == hf["fallback_model"]
    assert captured["gpu_mem"] == "4500MiB"
    assert captured["cpu_mem"] == "13GiB"


def test_build_backend_hf_respects_overrides_from_config(monkeypatch) -> None:
    cfg = load_config(load_secrets=False)
    cfg.raw["local"]["hf"]["gpu_mem"] = "9000MiB"
    cfg.raw["local"]["hf"]["cpu_mem"] = "30GiB"

    import decoding_sandbox.backends.hf as hf_mod

    captured: dict = {}

    class _Stub:
        def __init__(self, model_id, *, fallback_model=None, load_in_4bit=True,
                     gpu_mem="0MiB", cpu_mem="0MiB"):
            captured.update(gpu_mem=gpu_mem, cpu_mem=cpu_mem)

    monkeypatch.setattr(hf_mod, "HFBackend", _Stub)
    factory_mod.build_backend("hf", cfg)

    assert captured["gpu_mem"] == "9000MiB"
    assert captured["cpu_mem"] == "30GiB"


def test_build_backend_llamacpp_py_routes_with_normalized_name(monkeypatch) -> None:
    """Both ``llamacpp-py`` and ``llamacpp_py`` map to LlamaCppPyBackend, and
    every config field is forwarded to the constructor."""
    cfg = load_config(load_secrets=False)
    cfg.raw["local"]["llamacpp_py"]["model_path"] = "/tmp/explicit.gguf"
    cfg.raw["local"]["llamacpp_py"]["n_gpu_layers"] = 30
    cfg.raw["local"]["llamacpp_py"]["n_ctx"] = 2048
    cfg.raw["local"]["llamacpp_py"]["logits_all"] = True

    import decoding_sandbox.backends.llamacpp_py as lp_mod

    captured: dict = {}

    class _Stub:
        def __init__(
            self,
            *,
            model_path,
            model_glob,
            model_search_dirs,
            n_gpu_layers,
            n_ctx,
            logits_all,
            verbose,
        ):
            captured.update(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                logits_all=logits_all,
                verbose=verbose,
                model_glob=model_glob,
                model_search_dirs=model_search_dirs,
            )

    monkeypatch.setattr(lp_mod, "LlamaCppPyBackend", _Stub)

    for alias in ("llamacpp-py", "llamacpp_py", "llamacpp-python", "llama-py"):
        captured.clear()
        factory_mod.build_backend(alias, cfg)
        assert captured["model_path"] == "/tmp/explicit.gguf"
        assert captured["n_gpu_layers"] == 30
        assert captured["n_ctx"] == 2048
        assert captured["logits_all"] is True

    # CLI --model overrides model_path.
    captured.clear()
    factory_mod.build_backend("llamacpp-py", cfg, model="/tmp/other.gguf")
    assert captured["model_path"] == "/tmp/other.gguf"


def test_build_backend_routes_provider_to_openai_compat(monkeypatch) -> None:
    """Any name matching a provider should construct OpenAICompatBackend."""
    cfg = load_config(load_secrets=False)
    import decoding_sandbox.backends.openai_compat as oc

    seen: dict = {}

    class _Stub:
        def __init__(self, provider, model=None):
            seen["name"] = provider.name

    monkeypatch.setattr(oc, "OpenAICompatBackend", _Stub)
    for name in ("fireworks", "nim", "openrouter", "lmstudio"):
        factory_mod.build_backend(name, cfg)
        assert seen["name"] == name
