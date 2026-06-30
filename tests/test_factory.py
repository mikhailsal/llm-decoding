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
        def __init__(self, provider, model=None, **_kwargs):
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
        def __init__(self, base_url, **_kwargs):
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
        def __init__(
            self,
            model_id,
            *,
            fallback_model=None,
            load_in_4bit=True,
            gpu_mem="0MiB",
            cpu_mem="0MiB",
        ):
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
        def __init__(
            self,
            model_id,
            *,
            fallback_model=None,
            load_in_4bit=True,
            gpu_mem="0MiB",
            cpu_mem="0MiB",
        ):
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
        def __init__(self, provider, model=None, **_kwargs):
            seen["name"] = provider.name

    monkeypatch.setattr(oc, "OpenAICompatBackend", _Stub)
    for name in ("fireworks", "nim", "openrouter", "lmstudio"):
        factory_mod.build_backend(name, cfg)
        assert seen["name"] == name


# --------------------------------------------------------------------------- #
# Remote backend wiring
# --------------------------------------------------------------------------- #
def test_build_backend_routes_remote_alias(monkeypatch) -> None:
    """A name matching ``[remote.NAME]`` builds a RemoteBackend with the
    block's base_url + timeout."""
    from decoding_sandbox.core.config import RemoteConfig

    cfg = load_config(load_secrets=False)
    cfg.remotes = {
        "dsbx-host-py": RemoteConfig("dsbx-host-py", "http://dsbx-host:8000", timeout=42.0),
    }
    import decoding_sandbox.backends.remote as rmod

    captured: dict = {}

    class _Stub:
        def __init__(self, base_url, *, timeout=120.0, **_kwargs):
            captured["base_url"] = base_url
            captured["timeout"] = timeout

    monkeypatch.setattr(rmod, "RemoteBackend", _Stub)
    factory_mod.build_backend("dsbx-host-py", cfg)
    assert captured["base_url"] == "http://dsbx-host:8000"
    assert captured["timeout"] == 42.0


def test_build_backend_bare_remote_picks_single_alias(monkeypatch) -> None:
    """``--backend remote`` with exactly one [remote.NAME] entry picks it."""
    from decoding_sandbox.core.config import RemoteConfig

    cfg = load_config(load_secrets=False)
    cfg.remotes = {"only": RemoteConfig("only", "http://x:1")}
    import decoding_sandbox.backends.remote as rmod

    captured: dict = {}
    monkeypatch.setattr(
        rmod,
        "RemoteBackend",
        lambda base_url, *, timeout=120.0, **_kwargs: captured.update(base_url=base_url),
    )
    factory_mod.build_backend("remote", cfg)
    assert captured["base_url"] == "http://x:1"


def test_build_backend_bare_remote_errors_when_no_entries() -> None:
    """``--backend remote`` with no configured aliases is a helpful error."""
    cfg = load_config(load_secrets=False)
    cfg.remotes = {}
    with pytest.raises(ValueError, match=r"no \[remote\.NAME\] blocks"):
        factory_mod.build_backend("remote", cfg)


def test_build_backend_bare_remote_errors_when_ambiguous() -> None:
    """``--backend remote`` with multiple aliases asks the user to pick one."""
    from decoding_sandbox.core.config import RemoteConfig

    cfg = load_config(load_secrets=False)
    cfg.remotes = {
        "a": RemoteConfig("a", "http://a:1"),
        "b": RemoteConfig("b", "http://b:1"),
    }
    with pytest.raises(ValueError, match="ambiguous"):
        factory_mod.build_backend("remote", cfg)


def test_build_backend_unknown_lists_remotes_in_available_message() -> None:
    """The 'Backend not available' error should list configured remotes
    so users discover the right name without grepping config files."""
    from decoding_sandbox.core.config import RemoteConfig

    cfg = load_config(load_secrets=False)
    cfg.remotes = {"dsbx-host-py": RemoteConfig("dsbx-host-py", "http://x:1")}
    with pytest.raises(ValueError, match="dsbx-host-py"):
        factory_mod.build_backend("nope", cfg)


# --------------------------------------------------------------------------- #
# list_available_models
# --------------------------------------------------------------------------- #
def test_list_available_models_hf_collects_models_and_fallback() -> None:
    cfg = load_config(load_secrets=False)
    cfg.raw["local"]["hf"] = {
        "model": "Q/Q1",
        "fallback_model": "Q/Q2",
        "models": ["Q/Q3", "Q/Q1"],  # duplicate should be deduped
    }
    out = factory_mod.list_available_models("hf", cfg)
    assert out == [("Q/Q3", "Q/Q3"), ("Q/Q1", "Q/Q1"), ("Q/Q2", "Q/Q2")]


def test_list_available_models_llamacpp_py_discovers_gguf(monkeypatch) -> None:
    cfg = load_config(load_secrets=False)
    cfg.raw["local"]["llamacpp_py"] = {
        "model_search_dirs": ["/tmp/ggufs"],
        "model_path": "/tmp/explicit.gguf",
    }

    def fake_discover(dirs):
        return [("/tmp/a.gguf", "a"), ("/tmp/b.gguf", "b")]

    monkeypatch.setattr("decoding_sandbox.backends.llamacpp_py.discover_gguf_models", fake_discover)
    out = factory_mod.list_available_models("llamacpp-py", cfg)
    # explicit model_path is prepended when not already present
    assert out[0] == ("/tmp/explicit.gguf", "explicit")
    assert out[1:] == [("/tmp/a.gguf", "a"), ("/tmp/b.gguf", "b")]


def test_list_available_models_llamacpp_py_skips_duplicate_explicit(monkeypatch) -> None:
    cfg = load_config(load_secrets=False)
    cfg.raw["local"]["llamacpp_py"] = {
        "model_search_dirs": ["/tmp/ggufs"],
        "model_path": "/tmp/a.gguf",  # already returned by discover
    }

    def fake_discover(dirs):
        return [("/tmp/a.gguf", "a"), ("/tmp/b.gguf", "b")]

    monkeypatch.setattr("decoding_sandbox.backends.llamacpp_py.discover_gguf_models", fake_discover)
    out = factory_mod.list_available_models("llamacpp-py", cfg)
    # explicit path not duplicated
    assert out == [("/tmp/a.gguf", "a"), ("/tmp/b.gguf", "b")]


def test_list_available_models_unknown_backend_returns_empty() -> None:
    cfg = load_config(load_secrets=False)
    assert factory_mod.list_available_models("nope", cfg) == []
