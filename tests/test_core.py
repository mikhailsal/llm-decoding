from __future__ import annotations

import json
import math
import random

import pytest

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.engine import generate
from decoding_sandbox.core.manual import ManualSession
from decoding_sandbox.core.samplers import Sampler, SamplerContext
from decoding_sandbox.core.speculative import speculative_generate
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate


class FakeBackend(Backend):
    def __init__(
        self,
        *,
        tokens: dict[str, list[int]] | None = None,
        pieces: dict[int, str] | None = None,
        distributions: dict[tuple[int, ...], list[TokenCandidate]] | None = None,
        full_vocab: bool = True,
        prompt_logprobs: bool = True,
    ) -> None:
        self.tokens = tokens or {}
        self.pieces = pieces or {}
        self.distributions = distributions or {}
        self.full_vocab = full_vocab
        self.prompt_logprobs = prompt_logprobs
        self.closed = False

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="fake",
            full_vocab=self.full_vocab,
            prompt_logprobs=self.prompt_logprobs,
            max_top_logprobs=10,
            can_force_token=True,
        )

    def tokenize(self, text: str) -> list[int]:
        if text in self.tokens:
            return list(self.tokens[text])
        return [ord(ch) for ch in text]

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(self.piece(tid) for tid in token_ids)

    def piece(self, token_id: int) -> str:
        return self.pieces.get(token_id, chr(token_id))

    def next_distribution(self, token_ids: list[int], top_k: int) -> StepResult:
        cands = list(self.distributions.get(tuple(token_ids), []))[:top_k]
        return StepResult(position=len(token_ids), candidates=cands, is_full_vocab=self.full_vocab)

    def close(self) -> None:
        self.closed = True


def cand(token_id: int, text: str, prob: float, rank: int) -> TokenCandidate:
    return TokenCandidate(token_id, text, math.log(prob), rank)


def test_score_prompt_marks_actual_token_unknown_when_outside_top_k() -> None:
    backend = FakeBackend(
        tokens={"AB": [1, 2]},
        pieces={1: "A", 2: "B", 3: "C", 4: "D"},
        distributions={(1,): [cand(3, "C", 0.7, 0), cand(4, "D", 0.2, 1)]},
        full_vocab=False,
        prompt_logprobs=False,
    )

    [step] = backend.score_prompt("AB", top_k=2, watch_ids=[4])

    assert step.position == 1
    assert step.context_text == "A"
    assert step.chosen is not None
    assert step.chosen.token_id == 2
    assert step.chosen.rank == -1
    assert math.isnan(step.chosen.logprob)
    assert step.watched[4].text == "D"
    assert step.watched[4].rank == 1


def test_manual_session_pick_force_undo_and_save_load(tmp_path) -> None:
    backend = FakeBackend(
        tokens={"P": [10], " forced": [20, 21]},
        pieces={10: "P", 11: "!", 20: " forced", 21: "."},
        distributions={(10,): [cand(11, "!", 0.9, 0)]},
    )
    session = ManualSession(backend, "P", top_k=3)

    picked = session.pick(0)
    forced = session.force_text(" forced")
    undone = session.undo()

    assert picked.token_id == 11
    assert [c.token_id for c in forced] == [20, 21]
    assert undone == 21
    assert session.generated_ids == [11, 20]

    path = tmp_path / "transcript.json"
    session.save(path)
    raw = json.loads(path.read_text())
    assert raw["generated_ids"] == [11, 20]

    restored = ManualSession(backend, "placeholder")
    restored.load(path)
    assert restored.prompt == "P"
    assert restored.prompt_ids == [10]
    assert restored.generated_ids == [11, 20]


def test_generate_stops_after_stop_id() -> None:
    backend = FakeBackend(
        tokens={"P": [1]},
        pieces={1: "P", 2: "X", 3: "Y"},
        distributions={
            (1,): [cand(2, "X", 0.8, 0)],
            (1, 2): [cand(3, "Y", 0.8, 0)],
        },
    )
    sampler = Sampler("greedy", temperature=0.0)

    steps = list(generate(backend, "P", sampler, max_tokens=5, stop_ids=[2]))

    assert len(steps) == 1
    assert steps[0].decision.token_id == 2


def test_sampler_top_p_keeps_at_least_first_candidate() -> None:
    sampler = Sampler("top_p", top_p=0.0)
    cands = [cand(1, "A", 0.7, 0), cand(2, "B", 0.2, 1), cand(3, "C", 0.1, 2)]
    ctx = SamplerContext(step=0, token_ids=[], rng=random.Random(0))

    decision = sampler(cands, ctx)

    assert decision.token_id == 1
    assert [(c.token_id, p) for c, p in decision.kept] == [(1, 1.0)]


class DraftBackend(FakeBackend):
    def next_distribution(self, token_ids: list[int], top_k: int) -> StepResult:
        next_id = 100 + len(token_ids)
        return StepResult(
            position=len(token_ids),
            candidates=[TokenCandidate(next_id, f"T{next_id}", math.log(0.9), 0)],
            is_full_vocab=False,
        )


class TargetBackend(FakeBackend):
    def verify_greedy(
        self, context_ids: list[int], draft_ids: list[int]
    ) -> tuple[int, TokenCandidate]:
        return len(draft_ids), TokenCandidate(999, "bonus", math.log(0.9), 0)


def test_speculative_generate_never_exceeds_max_tokens() -> None:
    target = TargetBackend(tokens={"P": [1]}, pieces={1: "P"})
    draft = DraftBackend(tokens={"P": [1]}, pieces={1: "P"})

    rounds = list(speculative_generate(target, draft, "P", gamma=4, max_tokens=2))

    assert len(rounds) == 1
    assert len(rounds[0].emitted_ids) == 2
    assert rounds[0].emitted_ids == [101, 102]


def test_manual_pick_rejects_out_of_range_rank() -> None:
    backend = FakeBackend(tokens={"P": [1]}, distributions={(1,): []})
    session = ManualSession(backend, "P")

    with pytest.raises(IndexError, match="rank 0 out of range"):
        session.pick(0)
