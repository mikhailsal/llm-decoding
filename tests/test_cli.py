from __future__ import annotations

import argparse
import io
import math

import pytest
from rich.console import Console

from decoding_sandbox.cli import app
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import load_config
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate


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


def test_inspect_generated_only_provider_shows_next_token_distribution(monkeypatch) -> None:
    backend = OpenAICompatBackend()

    def fake_build_backend(name, cfg, model=None):
        assert name == "nim"
        return backend

    monkeypatch.setattr("decoding_sandbox.core.factory.build_backend", fake_build_backend)
    output = io.StringIO()
    old_console = app.console
    app.console = Console(file=output, force_terminal=False, color_system=None, width=120)
    try:
        rc = app.cmd_inspect(
            argparse.Namespace(
                backend="nim",
                model=None,
                prompt="Hello",
                top_k=2,
                watch=[],
                candidates=0,
            ),
            load_config(load_secrets=False),
        )
    finally:
        app.console = old_console

    rendered = output.getvalue()
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
                backend="unknown",
                model=None,
                prompt="Hello",
                top_k=2,
                watch=[],
                candidates=0,
            ),
            load_config(load_secrets=False),
        )
