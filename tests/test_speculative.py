"""Tests for the Speculator Protocol + HFSpeculator + speculative_generate driver."""

from __future__ import annotations

import math

import pytest

from dsbx.core.speculative import (
    HFSpeculator,
    Speculator,
    speculative_generate,
)
from dsbx.core.types import TokenCandidate
from tests.fakes import FakeBackend


class _SequentialDraft(FakeBackend):
    """Always proposes the next consecutive int id after the context."""

    def next_distribution(self, token_ids, top_k, *, watch_ids=()):
        from dsbx.core.types import StepResult

        next_id = 100 + len(token_ids)
        return StepResult(
            position=len(token_ids),
            candidates=[TokenCandidate(next_id, f"T{next_id}", math.log(0.9), 0)],
            is_full_vocab=False,
        )


class _AcceptAllTarget(FakeBackend):
    """Target that approves every draft and yields a fixed bonus token."""

    def verify_greedy(self, context_ids, draft_ids):
        return len(draft_ids), TokenCandidate(999, "bonus", math.log(0.9), 0)


class _PartialAcceptTarget(FakeBackend):
    """Target that accepts the first ``accept_n`` drafts then corrects."""

    def __init__(self, *, accept_n: int, correction_id: int, **kw) -> None:
        super().__init__(**kw)
        self.accept_n = accept_n
        self.correction_id = correction_id

    def verify_greedy(self, context_ids, draft_ids):
        acc = min(self.accept_n, len(draft_ids))
        corr = TokenCandidate(self.correction_id, f"R{self.correction_id}", math.log(0.7), 0)
        return acc, corr


class _NoBonusTarget(FakeBackend):
    """Target that accepts everything but has no further token to emit."""

    def verify_greedy(self, context_ids, draft_ids):
        return len(draft_ids), None


# --------------------------------------------------------------------------- #
# Backwards-compatible (target, draft) call form
# --------------------------------------------------------------------------- #
def test_speculative_generate_never_exceeds_max_tokens() -> None:
    target = _AcceptAllTarget(tokens={"P": [1]}, pieces={1: "P"})
    draft = _SequentialDraft(tokens={"P": [1]}, pieces={1: "P"})

    rounds = list(speculative_generate(target, draft, "P", gamma=4, max_tokens=2))

    assert len(rounds) == 1
    assert len(rounds[0].emitted_ids) == 2
    assert rounds[0].emitted_ids == [101, 102]


def test_speculative_generate_partial_accept_emits_correction() -> None:
    target = _PartialAcceptTarget(accept_n=2, correction_id=777, tokens={"P": [1]}, pieces={1: "P"})
    draft = _SequentialDraft(tokens={"P": [1]}, pieces={1: "P"})

    rounds = list(speculative_generate(target, draft, "P", gamma=4, max_tokens=5))

    assert rounds, "expected at least one round"
    first = rounds[0]
    assert first.accepted == 2
    assert first.correction is not None
    assert first.correction.token_id == 777
    # 2 accepted draft ids (101, 102) + correction (777)
    assert first.emitted_ids[:3] == [101, 102, 777]
    rejected_ids = [c.token_id for c in first.rejected]
    assert rejected_ids and rejected_ids[0] == 103


def test_speculative_round_rejected_property_exposes_unmatched_drafts() -> None:
    target = _PartialAcceptTarget(accept_n=1, correction_id=42, tokens={"P": [1]}, pieces={1: "P"})
    draft = _SequentialDraft(tokens={"P": [1]}, pieces={1: "P"})

    rounds = list(speculative_generate(target, draft, "P", gamma=3, max_tokens=2))

    rejected = rounds[0].rejected
    assert [c.token_id for c in rejected] == [102, 103]


def test_speculative_generate_breaks_when_no_token_emitted() -> None:
    target = _NoBonusTarget(tokens={"P": [1]}, pieces={1: "P"})

    class _EmptyDraft(FakeBackend):
        def next_distribution(self, token_ids, top_k, *, watch_ids=()):
            from dsbx.core.types import StepResult

            return StepResult(position=len(token_ids), candidates=[], is_full_vocab=False)

    rounds = list(
        speculative_generate(target, _EmptyDraft(tokens={"P": [1]}), "P", gamma=3, max_tokens=10)
    )

    assert len(rounds) == 1
    assert rounds[0].emitted_ids == []


def test_speculative_generate_truncates_emitted_to_remaining_budget() -> None:
    target = _AcceptAllTarget(tokens={"P": [1]}, pieces={1: "P"})
    draft = _SequentialDraft(tokens={"P": [1]}, pieces={1: "P"})

    rounds = list(speculative_generate(target, draft, "P", gamma=5, max_tokens=3))

    total_emitted = sum(len(r.emitted_ids) for r in rounds)
    assert total_emitted == 3


# --------------------------------------------------------------------------- #
# Speculator Protocol + HFSpeculator
# --------------------------------------------------------------------------- #
def test_hf_speculator_satisfies_protocol() -> None:
    target = _AcceptAllTarget(tokens={"P": [1]})
    draft = _SequentialDraft(tokens={"P": [1]})

    spec = HFSpeculator(target=target, draft=draft)

    assert isinstance(spec, Speculator)
    assert spec.target is target
    assert spec.draft is draft


def test_hf_speculator_propose_walks_draft_greedily() -> None:
    draft = _SequentialDraft(tokens={"P": [1]})
    target = _AcceptAllTarget(tokens={"P": [1]})
    spec = HFSpeculator(target=target, draft=draft)

    proposed = spec.propose([1], gamma=3)

    assert [c.token_id for c in proposed] == [101, 102, 103]


def test_hf_speculator_verify_routes_to_target_method() -> None:
    target = _PartialAcceptTarget(accept_n=1, correction_id=500, tokens={"P": [1]})
    draft = _SequentialDraft(tokens={"P": [1]})
    spec = HFSpeculator(target=target, draft=draft)

    accepted, correction = spec.verify([1], [101, 102, 103])

    assert accepted == 1
    assert correction is not None
    assert correction.token_id == 500


def test_hf_speculator_verify_requires_verify_greedy_on_target() -> None:
    target = FakeBackend(tokens={"P": [1]})  # no verify_greedy
    draft = _SequentialDraft(tokens={"P": [1]})
    spec = HFSpeculator(target=target, draft=draft)

    with pytest.raises(TypeError, match="verify_greedy"):
        spec.verify([1], [42])


def test_speculative_generate_accepts_speculator_directly() -> None:
    target = _AcceptAllTarget(tokens={"P": [1]})
    draft = _SequentialDraft(tokens={"P": [1]})
    spec = HFSpeculator(target=target, draft=draft)

    rounds = list(speculative_generate(spec, prompt="P", gamma=2, max_tokens=2))

    assert rounds
    assert rounds[0].emitted_ids == [101, 102]


def test_speculative_generate_requires_draft_when_passing_backend() -> None:
    target = _AcceptAllTarget(tokens={"P": [1]})

    with pytest.raises(TypeError, match="Speculator or both"):
        list(speculative_generate(target, prompt="P", gamma=2, max_tokens=2))
