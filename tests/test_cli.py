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


# --------------------------------------------------------------------------- #
# Watch target resolution (--watch / --watch-id / --watch-eos)
# --------------------------------------------------------------------------- #
def test_resolve_watch_text_uses_first_id_and_quoted_label() -> None:
    """The text-mode watch tokenizes and labels via the user's repr.
    The column header in the table is built off this label, so its
    quoted form is part of the public contract here."""
    backend = FakeBackend(
        tokens={" Paris": [42]},
        pieces={42: " Paris"},
    )
    [t] = app._resolve_watch(backend, [" Paris"])
    assert t.token_id == 42
    assert t.label.startswith("text:")
    assert "' Paris'" in t.label  # repr quoting preserved


def test_resolve_watch_skips_empty_tokenization(captured_console) -> None:
    """Tokens that yield no ids get a warning and are dropped (otherwise we'd
    have a column we cannot populate)."""
    backend = FakeBackend(tokens={"x": []})
    out = app._resolve_watch(backend, ["x"])
    assert out == []
    assert "tokenizes to nothing" in captured_console.getvalue()


def test_resolve_watch_id_includes_piece_in_label() -> None:
    backend = FakeBackend(pieces={1234: " Paris"})
    [t] = app._resolve_watch_ids(backend, [1234])
    assert t.token_id == 1234
    assert t.label.startswith("id=1234")
    assert "Paris" in t.label  # the piece is appended for context


def test_resolve_watch_id_handles_empty_piece() -> None:
    """A control token whose piece is empty still gets a clean label
    (the renderer's <special> marker takes over for the suffix)."""
    backend = FakeBackend(pieces={9999: ""})
    [t] = app._resolve_watch_ids(backend, [9999])
    assert t.label == "id=9999"


def test_resolve_watch_id_rejects_non_integer(captured_console) -> None:
    backend = FakeBackend()
    out = app._resolve_watch_ids(backend, ["nope"])  # type: ignore[arg-type]
    assert out == []
    assert "not an integer" in captured_console.getvalue()


def test_resolve_watch_eos_expands_from_capabilities() -> None:
    """--watch-eos pulls ids straight from capabilities so we don't depend on
    the text round-trip; each id becomes its own WatchTarget."""
    backend = FakeBackend(
        pieces={250: "", 251: ""},
        eos_token_ids=(250, 251),
    )
    out = app._resolve_watch_eos(backend)
    assert [t.token_id for t in out] == [250, 251]
    assert all(t.label.startswith("EOS:") for t in out)


def test_resolve_watch_eos_warns_when_unavailable(captured_console) -> None:
    """The HTTP llama.cpp / cloud-provider case: capabilities.eos_token_ids
    is empty, so --watch-eos must surface that explicitly instead of
    silently producing nothing."""
    backend = FakeBackend()  # default eos_token_ids=()
    assert app._resolve_watch_eos(backend) == []
    assert "does not expose EOS ids" in captured_console.getvalue()


def test_collect_watch_targets_dedups_across_sources() -> None:
    """When the same id arrives from --watch ' Paris' AND --watch-id 42,
    we keep one column (the text label wins, since it came first)."""
    backend = FakeBackend(
        tokens={" Paris": [42]},
        pieces={42: " Paris"},
    )
    out = app._collect_watch_targets(
        backend, texts=[" Paris"], ids=[42], eos=False
    )
    assert len(out) == 1
    assert out[0].label.startswith("text:")
    assert out[0].token_id == 42


def test_collect_watch_targets_preserves_source_order() -> None:
    """Columns line up with the user's mental model: texts first, then ids,
    then EOS expansion."""
    backend = FakeBackend(
        tokens={"a": [1], "b": [2]},
        pieces={1: "a", 2: "b", 9: "", 250: ""},
        eos_token_ids=(250,),
    )
    out = app._collect_watch_targets(
        backend, texts=["a", "b"], ids=[9], eos=True
    )
    assert [t.token_id for t in out] == [1, 2, 9, 250]


def test_collect_watch_targets_combines_text_id_and_eos() -> None:
    """The motivating real-world recipe: track ' Paris' AND EOS at every
    position in a fixed context."""
    backend = FakeBackend(
        tokens={" Paris": [42]},
        pieces={42: " Paris", 250: ""},
        eos_token_ids=(250,),
    )
    out = app._collect_watch_targets(
        backend, texts=[" Paris"], ids=[], eos=True
    )
    assert [t.token_id for t in out] == [42, 250]
    assert out[0].label.startswith("text:")
    assert out[1].label == "EOS:250"


# --------------------------------------------------------------------------- #
# Argument parser: the new --watch-id / --watch-eos flags
# --------------------------------------------------------------------------- #
def test_parser_inspect_collects_repeated_watch_id_flags() -> None:
    parser = app.build_parser()
    args = parser.parse_args(
        ["inspect", "hi", "--watch-id", "10", "--watch-id", "20"]
    )
    assert args.watch_id == [10, 20]


def test_parser_inspect_rejects_non_int_watch_id() -> None:
    parser = app.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["inspect", "hi", "--watch-id", "abc"])


def test_parser_inspect_watch_eos_defaults_false() -> None:
    parser = app.build_parser()
    args = parser.parse_args(["inspect", "hi"])
    assert args.watch_eos is False


def test_parser_inspect_watch_eos_flag_sets_true() -> None:
    parser = app.build_parser()
    args = parser.parse_args(["inspect", "hi", "--watch-eos"])
    assert args.watch_eos is True


# --------------------------------------------------------------------------- #
# End-to-end: --watch-eos renders an EOS column populated by score_prompt
# --------------------------------------------------------------------------- #
def test_inspect_watch_eos_renders_column_with_per_position_prob(
    monkeypatch, captured_console
) -> None:
    """The full path: --watch-eos collects the EOS id from capabilities,
    score_prompt threads it through watch_ids, the table renders a column
    labeled ``watch EOS:<id>`` and each row has a percentage for it."""
    backend = FakeBackend(
        tokens={"hi": [1, 2]},
        pieces={1: "h", 2: "i", 250: ""},
        eos_token_ids=(250,),
    )
    # FakeBackend.score_prompt isn't overridden, so emulate it directly via
    # a thin override on this instance.
    def fake_score(prompt, top_k, watch_ids=None):
        watch_ids = watch_ids or []
        watched = {
            wid: TokenCandidate(wid, backend.piece(wid), math.log(0.05), 7)
            for wid in watch_ids
        }
        return [
            StepResult(
                position=1, candidates=[cand(2, "i", 0.9, 0)], is_full_vocab=True,
                chosen=cand(2, "i", 0.9, 0), context_text="h", watched=watched,
            )
        ]
    backend.score_prompt = fake_score  # type: ignore[method-assign]
    monkeypatch.setattr(
        "decoding_sandbox.core.factory.build_backend", lambda *a, **kw: backend
    )

    rc = app.cmd_inspect(
        argparse.Namespace(
            backend="fake", model=None, prompt="hi", top_k=2,
            watch=[], watch_id=[], watch_eos=True,
            candidates=0, skip_preflight=True,
        ),
        _cfg_no_preflight(),
    )

    rendered = captured_console.getvalue()
    assert rc == 0
    assert "EOS:250" in rendered  # the new column header
    assert "5.00%" in rendered    # the per-position probability we faked
