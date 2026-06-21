"""Tests for config loading, secrets env-file handling, and provider lookup."""

from __future__ import annotations

import os

import pytest

from decoding_sandbox.core import config as cfg_mod


def test_deep_merge_overrides_scalar_and_recurses_into_dicts() -> None:
    base = {
        "a": 1,
        "b": {"x": 1, "y": {"deep": "old"}},
        "c": [1, 2],
    }
    override = {
        "a": 2,
        "b": {"y": {"deep": "new"}, "z": "added"},
        "c": [3],
    }

    merged = cfg_mod._deep_merge(base, override)

    assert merged["a"] == 2
    assert merged["b"]["x"] == 1
    assert merged["b"]["y"]["deep"] == "new"
    assert merged["b"]["z"] == "added"
    assert merged["c"] == [3]  # lists are replaced, not concatenated
    # Original base is not mutated:
    assert base["a"] == 1
    assert base["b"]["y"]["deep"] == "old"


def test_load_env_file_sets_missing_keys_but_does_not_overwrite(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / "secrets.env"
    env_path.write_text(
        "# comment\n"
        "\n"
        "EXISTING_KEY=should_not_overwrite\n"
        'NEW_KEY="quoted value"\n'
        "ANOTHER='single-quoted'\n"
        "NOEQUALS_LINE\n"  # ignored
    )
    monkeypatch.setenv("EXISTING_KEY", "keep_me")
    monkeypatch.delenv("NEW_KEY", raising=False)
    monkeypatch.delenv("ANOTHER", raising=False)

    n = cfg_mod.load_env_file(env_path)

    assert n == 2  # NEW_KEY and ANOTHER, but not EXISTING_KEY
    assert os.environ["EXISTING_KEY"] == "keep_me"
    assert os.environ["NEW_KEY"] == "quoted value"
    assert os.environ["ANOTHER"] == "single-quoted"


def test_load_env_file_returns_zero_for_missing_file(tmp_path) -> None:
    n = cfg_mod.load_env_file(tmp_path / "nope.env")
    assert n == 0


def test_load_config_uses_defaults_when_no_file(tmp_path, monkeypatch) -> None:
    # Force discovery to find nothing (point REPO_ROOT to an empty dir).
    monkeypatch.setattr(cfg_mod, "REPO_ROOT", tmp_path)

    cfg = cfg_mod.load_config(load_secrets=False)

    assert cfg.config_path is None
    assert cfg.default_backend == "llamacpp"
    assert "fireworks" in cfg.providers
    assert cfg.providers["fireworks"].supports_prompt_logprobs is True


def test_load_config_merges_overrides_from_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cfg_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "config.toml").write_text('[run]\nbackend = "hf"\n[storage]\nmin_free_gb = 12.5\n')

    cfg = cfg_mod.load_config(load_secrets=False)

    assert cfg.default_backend == "hf"
    assert cfg.storage.min_free_gb == pytest.approx(12.5)


def test_load_config_explicit_path_raises_when_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        cfg_mod.load_config(tmp_path / "absent.toml")


def test_provider_lookup_raises_keyerror_for_unknown_name() -> None:
    cfg = cfg_mod.load_config(load_secrets=False)
    with pytest.raises(KeyError, match="Unknown provider"):
        cfg.provider("nope")


def test_provider_api_key_reads_from_env(monkeypatch) -> None:
    cfg = cfg_mod.load_config(load_secrets=False)
    monkeypatch.setenv(cfg.provider("fireworks").api_key_env, "test-key")
    assert cfg.provider("fireworks").api_key() == "test-key"


def test_config_get_returns_nested_value_or_default() -> None:
    cfg = cfg_mod.load_config(load_secrets=False)
    assert cfg.get("local", "hf", "model")  # exists
    assert cfg.get("local", "missing", default="fallback") == "fallback"
    assert cfg.get("local", "hf", "nope", default=None) is None


def test_expand_resolves_tilde(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/tmp/fakehome")
    p = cfg_mod.expand("~/x.txt")
    assert str(p) == "/tmp/fakehome/x.txt"


def test_expand_resolves_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("MYVAR", "/tmp/somewhere")
    p = cfg_mod.expand("$MYVAR/foo")
    assert str(p) == "/tmp/somewhere/foo"
