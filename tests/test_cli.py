"""Tests for the argparse + dispatch layer (no real backends or HTTP)."""

from __future__ import annotations

import argparse
import io
import math

import pytest
from rich.console import Console

from decoding_sandbox.cli import app
from decoding_sandbox.core import storage as storage_mod
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config, StorageConfig, load_config
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate
from tests.fakes import FakeBackend, cand


class OpenAICompatBackend(Backend):
    """Synthetic-token fake with the same class name as the real provider backend."""

    def __init__(self) -> None:
        self.score_prompt_called = False
        self.closed = False
        self._pieces = {0: "Hello", 1: " world", 2: " there"}

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="nim:fake",
            full_vocab=False,
            prompt_logprobs=False,
            max_top_logprobs=2,
            can_force_token=False,
            notes="chat-only top-k",
        )

    def tokenize(self, text: str) -> list[int]:
        return [0]

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(self.piece(tid) for tid in token_ids)

    def piece(self, token_id: int) -> str:
        return self._pieces.get(token_id, "")

    def next_distribution(self, token_ids: list[int], top_k: int) -> StepResult:
        assert token_ids == [0]
        return StepResult(
            position=1,
            candidates=[
                TokenCandidate(1, " world", math.log(0.75), 0),
                TokenCandidate(2, " there", math.log(0.25), 1),
            ][:top_k],
            is_full_vocab=False,
        )

    def score_prompt(
        self, prompt: str, top_k: int, watch_ids: list[int] | None = None
    ) -> list[StepResult]:
        self.score_prompt_called = True
        raise AssertionError("generated-token-only providers must not use prompt scoring")

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def captured_console(monkeypatch):
    output = io.StringIO()
    old = app.console
    app.console = Console(file=output, force_terminal=False, color_system=None, width=120)
    try:
        yield output
    finally:
        app.console = old


def _cfg_no_preflight() -> Config:
    cfg = load_config(load_secrets=False)
    cfg.storage = StorageConfig(
        hf_home="", pip_cache="", min_free_gb=0.0, check_paths=[]
    )
    return cfg


# --------------------------------------------------------------------------- #
# Inspect (chat-only fallback + invalid backend)
# --------------------------------------------------------------------------- #
def test_inspect_generated_only_provider_shows_next_token_distribution(
    monkeypatch, captured_console
) -> None:
    backend = OpenAICompatBackend()

    def fake_build_backend(name, cfg, model=None):
        assert name == "nim"
        return backend

    monkeypatch.setattr("decoding_sandbox.core.factory.build_backend", fake_build_backend)
    rc = app.cmd_inspect(
        argparse.Namespace(
            backend="nim", model=None, prompt="Hello", top_k=2,
            watch=[], candidates=0, skip_preflight=True,
        ),
        _cfg_no_preflight(),
    )

    rendered = captured_console.getvalue()
    assert rc == 0
    assert not backend.score_prompt_called
    assert backend.closed
    assert "cannot score prompt tokens" in rendered
    assert "Next-token inspection" in rendered
    assert "world" in rendered


def test_inspect_propagates_invalid_custom_backend(monkeypatch) -> None:
    def fake_build_backend(name, cfg, model=None):
        raise ValueError("bad backend")

    monkeypatch.setattr("decoding_sandbox.core.factory.build_backend", fake_build_backend)
    with pytest.raises(ValueError, match="bad backend"):
        app.cmd_inspect(
            argparse.Namespace(
                backend="unknown", model=None, prompt="Hello", top_k=2,
                watch=[], candidates=0, skip_preflight=True,
            ),
            _cfg_no_preflight(),
        )


def test_inspect_renders_watch_columns_in_chat_only_path(
    monkeypatch, captured_console
) -> None:
    backend = OpenAICompatBackend()
    monkeypatch.setattr(
        "decoding_sandbox.core.factory.build_backend", lambda *a, **kw: backend
    )

    rc = app.cmd_inspect(
        argparse.Namespace(
            backend="nim", model=None, prompt="Hello", top_k=2,
            watch=[" world"], candidates=0, skip_preflight=True,
        ),
        _cfg_no_preflight(),
    )

    assert rc == 0
    rendered = captured_console.getvalue()
    assert "watch" in rendered  # column header for watched tokens


# --------------------------------------------------------------------------- #
# Preflight wiring
# --------------------------------------------------------------------------- #
def test_run_preflight_returns_exit_code_on_low_disk(monkeypatch, captured_console) -> None:
    cfg = load_config(load_secrets=False)
    cfg.storage = StorageConfig(
        hf_home="", pip_cache="", min_free_gb=1.0, check_paths=["/anywhere"]
    )

    def boom(paths, min_free_gb):
        raise storage_mod.StoragePreflightError("simulated low disk")

    monkeypatch.setattr(storage_mod, "preflight_or_raise", boom)

    rc = app._run_preflight(cfg, skip=False)
    assert rc == 3
    assert "preflight failed" in captured_console.getvalue()


def test_run_preflight_skip_bypasses_check(monkeypatch) -> None:
    cfg = load_config(load_secrets=False)
    monkeypatch.setattr(
        storage_mod, "preflight_or_raise",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    assert app._run_preflight(cfg, skip=True) is None


def test_run_preflight_returns_none_on_ok(monkeypatch) -> None:
    cfg = load_config(load_secrets=False)
    monkeypatch.setattr(storage_mod, "preflight_or_raise", lambda paths, min_free_gb: [])
    assert app._run_preflight(cfg, skip=False) is None


def test_cmd_inspect_aborts_when_preflight_fails(monkeypatch) -> None:
    """cmd_inspect must NOT build a backend when preflight fails."""
    builder_called = False

    def fake_build_backend(name, cfg, model=None):
        nonlocal builder_called
        builder_called = True
        raise AssertionError("backend should never be built")

    monkeypatch.setattr("decoding_sandbox.core.factory.build_backend", fake_build_backend)
    cfg = load_config(load_secrets=False)
    cfg.storage = StorageConfig(
        hf_home="", pip_cache="", min_free_gb=1.0, check_paths=["/anywhere"]
    )
    monkeypatch.setattr(
        storage_mod, "preflight_or_raise",
        lambda *a, **kw: (_ for _ in ()).throw(storage_mod.StoragePreflightError("nope")),
    )

    rc = app.cmd_inspect(
        argparse.Namespace(
            backend="hf", model=None, prompt="Hello", top_k=2,
            watch=[], candidates=0, skip_preflight=False,
        ),
        cfg,
    )

    assert rc == 3
    assert builder_called is False


# --------------------------------------------------------------------------- #
# Generate (--stop, --sampler)
# --------------------------------------------------------------------------- #
def test_cmd_generate_stops_on_resolved_stop_token(monkeypatch, captured_console) -> None:
    """End-to-end: cmd_generate halts the loop the step a stop id is chosen."""
    # The stop string "STOP" maps to id 99; after a single greedy step we
    # should see id 99 and exit. Without the wiring this would run 10 steps.
    backend = FakeBackend(
        tokens={"P": [1], "STOP": [99]},
        pieces={1: "P", 99: "STOP", 7: "x"},
        distributions={
            (1,): [cand(99, "STOP", 0.9, 0)],
            (1, 99): [cand(7, "x", 0.9, 0)],
        },
    )

    # Spy on generate() to count yielded steps.
    from decoding_sandbox.core import engine

    real_generate = engine.generate
    step_count = {"n": 0}

    def counting_generate(*a, **kw):
        for step in real_generate(*a, **kw):
            step_count["n"] += 1
            yield step

    monkeypatch.setattr(
        "decoding_sandbox.core.factory.build_backend", lambda *a, **kw: backend
    )
    monkeypatch.setattr("decoding_sandbox.cli.app.generate", counting_generate, raising=False)
    # ^ generate is imported *inside* cmd_generate, so we patch the engine module too.
    monkeypatch.setattr(engine, "generate", counting_generate)

    rc = app.cmd_generate(
        argparse.Namespace(
            backend="fake", model=None, prompt="P",
            sampler="greedy", custom_file=None,
            temperature=0.0, sampler_top_k=None, top_p=None, min_p=None, typical_p=None,
            max_tokens=10, seed=0, top_k=5,
            stop=["STOP"], skip_preflight=True,
        ),
        _cfg_no_preflight(),
    )

    assert rc == 0
    assert step_count["n"] == 1, (
        f"expected to stop after 1 step (stop id chosen), got {step_count['n']}"
    )
    rendered = captured_console.getvalue()
    assert "stop=" in rendered


def test_cmd_generate_warns_on_multi_token_stop(monkeypatch, captured_console) -> None:
    backend = FakeBackend(
        tokens={"P": [1], "AB": [2, 3]},  # multi-token stop
        pieces={1: "P", 2: "A", 3: "B"},
        distributions={(1,): [cand(2, "A", 0.9, 0)]},
    )
    monkeypatch.setattr(
        "decoding_sandbox.core.factory.build_backend", lambda *a, **kw: backend
    )

    rc = app.cmd_generate(
        argparse.Namespace(
            backend="fake", model=None, prompt="P",
            sampler="greedy", custom_file=None,
            temperature=0.0, sampler_top_k=None, top_p=None, min_p=None, typical_p=None,
            max_tokens=1, seed=0, top_k=5,
            stop=["AB"], skip_preflight=True,
        ),
        _cfg_no_preflight(),
    )
    assert rc == 0
    assert "cannot match per-step" in captured_console.getvalue()


def test_cmd_generate_custom_sampler_requires_file(captured_console) -> None:
    rc = app.cmd_generate(
        argparse.Namespace(
            backend=None, model=None, prompt="P",
            sampler="custom", custom_file=None,
            temperature=1.0, sampler_top_k=None, top_p=None, min_p=None, typical_p=None,
            max_tokens=1, seed=0, top_k=5,
            stop=[], skip_preflight=True,
        ),
        _cfg_no_preflight(),
    )
    assert rc == 2
    assert "requires --custom-file" in captured_console.getvalue()


def test_resolve_stop_ids_skips_empty_tokenization(monkeypatch, captured_console) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(backend, "tokenize", lambda text: [] if text == "EMPTY" else [1])

    resolved = app._resolve_stop_ids(backend, ["EMPTY", "OK"])

    assert resolved == [("OK", 1)]
    assert "tokenizes to nothing" in captured_console.getvalue()


# --------------------------------------------------------------------------- #
# Doctor command
# --------------------------------------------------------------------------- #
def test_cmd_doctor_renders_provider_and_storage_tables(monkeypatch, captured_console) -> None:
    cfg = load_config(load_secrets=False)
    cfg.storage = StorageConfig(
        hf_home="", pip_cache="", min_free_gb=5.0, check_paths=[]
    )
    rc = app.cmd_doctor(argparse.Namespace(), cfg)

    assert rc == 0
    rendered = captured_console.getvalue()
    assert "Provider API keys" in rendered
    assert "Storage" in rendered


def test_cmd_doctor_marks_lmstudio_as_no_key_needed(monkeypatch, captured_console) -> None:
    monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)
    cfg = load_config(load_secrets=False)
    cfg.storage = StorageConfig(
        hf_home="", pip_cache="", min_free_gb=5.0, check_paths=[]
    )
    app.cmd_doctor(argparse.Namespace(), cfg)
    rendered = captured_console.getvalue()
    assert "no key needed" in rendered


def test_cmd_doctor_returns_one_on_low_disk(monkeypatch, captured_console, tmp_path) -> None:
    cfg = load_config(load_secrets=False)
    cfg.storage = StorageConfig(
        hf_home="", pip_cache="", min_free_gb=10.0, check_paths=[str(tmp_path)]
    )

    class _Usage:
        total = 100 * storage_mod.GIB
        used = 99 * storage_mod.GIB
        free = 1 * storage_mod.GIB

    monkeypatch.setattr(storage_mod.shutil, "disk_usage", lambda p: _Usage)

    rc = app.cmd_doctor(argparse.Namespace(), cfg)

    assert rc == 1
    assert "LOW" in captured_console.getvalue()


def test_mask_no_key_ok_when_legitimately_keyless() -> None:
    out = app._mask(None, no_key_ok=True)
    assert "no key needed" in out


def test_mask_red_missing_when_required() -> None:
    out = app._mask(None, no_key_ok=False)
    assert "missing" in out


def test_mask_short_secret_hides_value() -> None:
    out = app._mask("abcd")
    assert "abcd" not in out
    assert "set" in out


def test_mask_long_secret_shows_prefix_and_suffix() -> None:
    out = app._mask("0123456789ABCDEF")
    assert "0123" in out
    assert "DEF" in out


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #
def test_build_parser_accepts_inspect_with_skip_preflight() -> None:
    parser = app.build_parser()
    args = parser.parse_args(["inspect", "hi", "--skip-preflight"])
    assert args.skip_preflight is True


def test_build_parser_default_skip_preflight_false() -> None:
    parser = app.build_parser()
    args = parser.parse_args(["inspect", "hi"])
    assert args.skip_preflight is False


def test_build_parser_generate_collects_repeated_stop_flags() -> None:
    parser = app.build_parser()
    args = parser.parse_args(["generate", "hi", "--stop", "A", "--stop", "B"])
    assert args.stop == ["A", "B"]


def test_build_parser_rejects_unknown_subcommand() -> None:
    parser = app.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["nope"])


# --------------------------------------------------------------------------- #
# main() entry point (smoke)
# --------------------------------------------------------------------------- #
def test_main_smoke_doctor_with_skip_preflight(monkeypatch, captured_console) -> None:
    """``dsbx doctor`` runs against built-in defaults without exploding."""
    # doctor doesn't actually go through _run_preflight, but it calls
    # storage.check_paths -- patch to skip real disks.
    monkeypatch.setattr(
        storage_mod, "check_paths",
        lambda paths, min_free_gb: [],
    )
    rc = app.main(["doctor"])
    assert rc == 0
    assert "Decoding Sandbox doctor" in captured_console.getvalue()


def test_main_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        app.main([])
