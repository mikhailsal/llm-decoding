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


def test_fireworks_defaults_carry_corrected_tokenizers_and_denylist(
    tmp_path, monkeypatch
) -> None:
    """The Fireworks defaults map each model to its authoritative HF repo
    and denylist the chat-only model.

    Pins: (a) ``minimax-m2p7`` resolves to MiniMax M2.7 (NOT the older
    MiniMax-M2 -- a wrong mapping would tokenize against the wrong vocab),
    (b) the per-release GLM repos are used, and (c) ``minimax-m3`` (chat
    only, hangs on /v1/completions) is on ``exclude_models``.
    """
    monkeypatch.setattr(cfg_mod, "REPO_ROOT", tmp_path)
    prov = cfg_mod.load_config(load_secrets=False).providers["fireworks"]
    toks = prov.tokenizers
    assert toks["accounts/fireworks/models/minimax-m2p7"] == "MiniMaxAI/MiniMax-M2.7"
    assert toks["accounts/fireworks/models/glm-5p1"] == "zai-org/GLM-5.1-FP8"
    assert toks["accounts/fireworks/models/glm-5p2"] == "zai-org/GLM-5.2"
    assert (
        toks["accounts/fireworks/models/deepseek-v4-pro"]
        == "deepseek-ai/DeepSeek-V4-Pro"
    )
    # Kimi (tiktoken, no tokenizer.json) and Qwen-plus (no public repo)
    # are intentionally absent from the tokenizer map.
    assert "accounts/fireworks/models/kimi-k2p6" not in toks
    assert "accounts/fireworks/models/qwen3p7-plus" not in toks
    # But they ARE offered in the picker (they support text completion).
    assert "accounts/fireworks/models/kimi-k2p6" in prov.models
    assert "accounts/fireworks/models/qwen3p7-plus" in prov.models
    # The chat-only model is denylisted.
    assert "accounts/fireworks/models/minimax-m3" in prov.exclude_models


def test_load_config_merges_overrides_from_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cfg_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "config.toml").write_text('[run]\nbackend = "hf"\n[storage]\nmin_free_gb = 12.5\n')

    cfg = cfg_mod.load_config(load_secrets=False)

    assert cfg.default_backend == "hf"
    assert cfg.storage.min_free_gb == pytest.approx(12.5)


def test_load_config_explicit_path_raises_when_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        cfg_mod.load_config(tmp_path / "absent.toml")


def test_load_config_pulls_in_repo_root_env_file(tmp_path, monkeypatch) -> None:
    """A ``<repo>/.env`` is loaded alongside ``secrets_env_file``.

    Without this contributors who park keys (chiefly ``HF_TOKEN``) in
    the project-local ``.env`` -- the standard idiom -- find that the
    web server silently ignores them, because only the central
    ``secrets_env_file`` is sourced. The fix loads BOTH (central
    first, then repo); ``load_env_file`` never overwrites existing
    values so the central file still wins when both define a key.
    """
    monkeypatch.setattr(cfg_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "config.toml").write_text("")
    (tmp_path / ".env").write_text("HF_TOKEN=hf_test_repo_local\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    cfg_mod.load_config(load_secrets=True)

    assert os.environ.get("HF_TOKEN") == "hf_test_repo_local"


def test_load_config_repo_env_does_not_overwrite_central(tmp_path, monkeypatch) -> None:
    """Central ``secrets_env_file`` wins when both define the same key."""
    monkeypatch.setattr(cfg_mod, "REPO_ROOT", tmp_path)
    central = tmp_path / "central.env"
    central.write_text("HF_TOKEN=central_wins\n")
    (tmp_path / "config.toml").write_text(
        f'secrets_env_file = "{central}"\n'
    )
    (tmp_path / ".env").write_text("HF_TOKEN=repo_loses\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    cfg_mod.load_config(load_secrets=True)

    assert os.environ["HF_TOKEN"] == "central_wins"


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


# --------------------------------------------------------------------------- #
# [remote.NAME] blocks
# --------------------------------------------------------------------------- #
def test_load_config_parses_remote_blocks(tmp_path, monkeypatch) -> None:
    """Each ``[remote.NAME]`` table becomes a RemoteConfig entry; bare
    ``remote.<name>.timeout`` overrides the default."""
    monkeypatch.setattr(cfg_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "config.toml").write_text(
        '[remote.dsbx-host-py]\n'
        'base_url = "http://192.0.2.42:8000"\n'
        'timeout = 42.0\n'
        '[remote.dsbx-host-hf]\n'
        'base_url = "http://192.0.2.42:8001"\n'
    )

    cfg = cfg_mod.load_config(load_secrets=False)

    assert set(cfg.remotes) == {"dsbx-host-py", "dsbx-host-hf"}
    assert cfg.remote("dsbx-host-py").base_url == "http://192.0.2.42:8000"
    assert cfg.remote("dsbx-host-py").timeout == pytest.approx(42.0)
    # Default timeout when omitted.
    assert cfg.remote("dsbx-host-hf").timeout == pytest.approx(120.0)


def test_load_config_rejects_remote_block_without_base_url(tmp_path, monkeypatch) -> None:
    """A ``[remote.foo]`` block missing ``base_url`` is a clear error
    rather than silently producing an unreachable entry."""
    monkeypatch.setattr(cfg_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "config.toml").write_text("[remote.foo]\ntimeout = 30\n")

    with pytest.raises(ValueError, match="base_url"):
        cfg_mod.load_config(load_secrets=False)


def test_remote_lookup_raises_keyerror_for_unknown_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cfg_mod, "REPO_ROOT", tmp_path)
    cfg = cfg_mod.load_config(load_secrets=False)
    with pytest.raises(KeyError, match="Unknown remote"):
        cfg.remote("nope")
