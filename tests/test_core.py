"""Tests for the pure-logic core (samplers, engine, manual session, types)."""

from __future__ import annotations

import json
import math
import random

import pytest

from decoding_sandbox.core.engine import generate
from decoding_sandbox.core.manual import ManualSession
from decoding_sandbox.core.samplers import Sampler, SamplerContext
from tests.fakes import FakeBackend, cand


# --------------------------------------------------------------------------- #
# Backend.score_prompt generic fallback
# --------------------------------------------------------------------------- #
def test_score_prompt_marks_actual_token_unknown_when_outside_top_k() -> None:
    """The base Backend.score_prompt returns one row per prompt token
    (N steps for an N-token prompt). The first N-1 score the actual next
    token; the last is the trailing "predict next" row with chosen=None.
    """
    backend = FakeBackend(
        tokens={"AB": [1, 2]},
        pieces={1: "A", 2: "B", 3: "C", 4: "D"},
        distributions={
            (1,): [cand(3, "C", 0.7, 0), cand(4, "D", 0.2, 1)],
            (1, 2): [cand(3, "C", 0.5, 0), cand(4, "D", 0.3, 1)],
        },
        full_vocab=False,
        prompt_logprobs=False,
    )

    scored, trailing = backend.score_prompt("AB", top_k=2, watch_ids=[4])

    assert scored.position == 1
    assert scored.context_text == "A"
    assert scored.chosen is not None
    assert scored.chosen.token_id == 2
    assert scored.chosen.rank == -1
    assert math.isnan(scored.chosen.logprob)
    assert scored.watched[4].text == "D"
    assert scored.watched[4].rank == 1

    assert trailing.position == 2
    assert trailing.context_text == "B"
    assert trailing.chosen is None  # no actual next token to verify
    assert trailing.watched[4].rank == 1  # still resolved from top-k


def test_score_prompt_watch_records_top_k_member_with_correct_rank() -> None:
    """Watch tokens that *are* within the top-k must report their rank/prob
    on every row, including the trailing prediction."""
    backend = FakeBackend(
        tokens={"AB": [1, 2]},
        pieces={1: "A", 2: "B", 3: "C", 4: "D"},
        distributions={
            (1,): [cand(3, "C", 0.7, 0), cand(4, "D", 0.2, 1)],
            (1, 2): [cand(3, "C", 0.6, 0), cand(4, "D", 0.3, 1)],
        },
        full_vocab=False,
        prompt_logprobs=False,
    )

    steps = backend.score_prompt("AB", top_k=2, watch_ids=[3])

    assert len(steps) == 2
    for st in steps:
        assert st.watched[3].rank == 0
        assert not math.isnan(st.watched[3].logprob)


def test_score_prompt_prepend_token_ids_adds_leading_bos_conditioned_row() -> None:
    """``prepend_token_ids`` lets the caller seed scoring with extra ids.

    Pedagogical use case: "predict position 0 from BOS". For the
    autoregressive model the user's first prompt token is normally
    unscorable (no prior to condition on); injecting the model's BOS
    in front gives the first scoring row a meaningful distribution.

    We pin the contract with a tiny FakeBackend whose distributions
    are unique per context so the prepended row can be unambiguously
    identified: after ``[bos=99]`` the model is asked to predict
    ``ids[0]`` of the (prepend + prompt) sequence, which is the BOS
    itself (the only "real" StepResult comparing against an actual
    next token between bos and the prompt is the row scoring the
    BOS's prediction of the first user token). When we DON'T pass
    ``prepend_token_ids`` we get the historical 2-row output for the
    2-token prompt; WITH a single prepended id we get 3 rows, and
    the leading row's ``context_text`` is the BOS piece while its
    ``chosen`` is the user's first prompt token with the
    BOS-conditioned probability. The trailing row still has
    ``chosen=None`` as before. Same fixture data, two assertion
    paths, so the diff between the two cases is exactly the leading
    BOS-conditioned row.
    """
    backend = FakeBackend(
        tokens={"AB": [1, 2]},
        pieces={1: "A", 2: "B", 3: "C", 99: "<BOS>"},
        distributions={
            (1,): [cand(2, "B", 0.9, 0)],
            (1, 2): [cand(3, "C", 0.8, 0)],
            (99,): [cand(1, "A", 0.4, 0), cand(2, "B", 0.3, 1)],
            (99, 1): [cand(2, "B", 0.85, 0)],
            (99, 1, 2): [cand(3, "C", 0.75, 0)],
        },
    )

    # Baseline: no prepend -> 2 rows, no BOS-conditioned leading row.
    plain = backend.score_prompt("AB", top_k=2)
    assert [s.position for s in plain] == [1, 2]
    assert plain[0].context_text == "A"  # context is the first user token

    # With prepend=[99]: 3 rows. The first row scores what the model
    # predicts AFTER seeing the BOS -- chosen is the user's first
    # prompt token (id=1, "A") with the BOS-conditioned probability
    # (0.4 from our fixture, NOT 0.0 or NaN -- this is the whole
    # point of the feature).
    with_bos = backend.score_prompt("AB", top_k=2, prepend_token_ids=[99])
    assert [s.position for s in with_bos] == [1, 2, 3]
    assert with_bos[0].context_text == "<BOS>"
    assert with_bos[0].chosen is not None
    assert with_bos[0].chosen.token_id == 1  # user's first prompt token
    assert with_bos[0].chosen.text == "A"
    assert with_bos[0].chosen.rank == 0  # top-1 under BOS context
    assert with_bos[0].chosen.logprob == pytest.approx(math.log(0.4))
    # Subsequent rows are the usual per-prompt-token rows but with
    # extended context (BOS + ids[:i]); the trailing row still has
    # chosen=None because there's nothing after the user's last token.
    assert with_bos[1].context_text == "A"
    assert with_bos[1].chosen is not None
    assert with_bos[1].chosen.token_id == 2
    assert with_bos[2].context_text == "B"
    assert with_bos[2].chosen is None


def test_score_prompt_trailing_step_has_chosen_none() -> None:
    """The trailing step's chosen=None is part of the public contract: it
    distinguishes "predicting" from "scoring against ground truth"."""
    backend = FakeBackend(
        tokens={"AB": [1, 2]},
        pieces={1: "A", 2: "B", 3: "C"},
        distributions={
            (1,): [cand(2, "B", 0.9, 0)],
            (1, 2): [cand(3, "C", 0.8, 0)],
        },
    )

    steps = backend.score_prompt("AB", top_k=1)

    assert [s.chosen is None for s in steps] == [False, True]
    # The trailing row still carries the model's top-1 prediction so
    # callers can render confidence / inspect P(<some-id>) at that
    # position.
    assert steps[-1].top is not None
    assert steps[-1].top.token_id == 3


def test_lookup_watch_returns_nan_candidate_for_missing_token() -> None:
    """The public lookup_watch helper marks unseen tokens as <top-k."""
    backend = FakeBackend(pieces={42: "x"})
    from decoding_sandbox.core.types import StepResult

    empty_step = StepResult(position=0, candidates=[], is_full_vocab=False)
    watch = backend.lookup_watch(empty_step, 42)

    assert watch.token_id == 42
    assert watch.text == "x"
    assert watch.rank == -1
    assert math.isnan(watch.logprob)


# --------------------------------------------------------------------------- #
# ManualSession
# --------------------------------------------------------------------------- #
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


def test_manual_pick_rejects_out_of_range_rank() -> None:
    backend = FakeBackend(tokens={"P": [1]}, distributions={(1,): []})
    session = ManualSession(backend, "P")

    with pytest.raises(IndexError, match="rank 0 out of range"):
        session.pick(0)


def test_manual_force_id_appends_and_marks_unknown_rank() -> None:
    backend = FakeBackend(tokens={"P": [1]}, pieces={99: "ZZ"})
    session = ManualSession(backend, "P")

    appended = session.force_id(99)

    assert session.generated_ids == [99]
    assert appended.rank == -1
    assert math.isnan(appended.logprob)
    assert appended.text == "ZZ"


def test_manual_undo_returns_none_when_nothing_to_undo() -> None:
    backend = FakeBackend(tokens={"P": [1]})
    session = ManualSession(backend, "P")

    assert session.undo() is None


def test_manual_full_text_concatenates_prompt_and_generated() -> None:
    backend = FakeBackend(
        tokens={"hi": [104, 105]},
        pieces={104: "h", 105: "i", 33: "!"},
        distributions={(104, 105): [cand(33, "!", 0.9, 0)]},
    )
    session = ManualSession(backend, "hi")
    session.pick(0)

    assert session.full_text() == "hi!"


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
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


def test_generate_records_changed_greedy_marker_for_runner_up_picker() -> None:
    """When a sampler picks a non-greedy candidate, decision.changed_greedy is True."""
    from decoding_sandbox.core.samplers import SamplerDecision

    backend = FakeBackend(
        tokens={"P": [1]},
        pieces={1: "P", 2: "X", 3: "Y"},
        distributions={(1,): [cand(2, "X", 0.7, 0), cand(3, "Y", 0.3, 1)]},
    )

    def sampler(cands, ctx):
        return SamplerDecision(
            token_id=cands[1].token_id,
            token_text=cands[1].text,
            kept=[(cands[0], 0.7), (cands[1], 0.3)],
            greedy_token_id=cands[0].token_id,
            note="runner-up",
        )

    steps = list(generate(backend, "P", sampler, max_tokens=1))

    assert len(steps) == 1
    assert steps[0].decision.changed_greedy is True
    assert steps[0].decision.token_id == 3


def test_generate_stops_when_distribution_is_empty() -> None:
    backend = FakeBackend(tokens={"P": [1]}, distributions={(1,): []})
    steps = list(generate(backend, "P", Sampler("greedy"), max_tokens=5))

    assert steps == []


def test_generate_chosen_candidate_helper_returns_kept_member() -> None:
    backend = FakeBackend(
        tokens={"P": [1]},
        distributions={(1,): [cand(2, "X", 0.9, 0)]},
    )
    [gs] = list(generate(backend, "P", Sampler("greedy", temperature=0.0), max_tokens=1))

    chosen = gs.chosen_candidate()
    assert chosen is not None
    assert chosen.token_id == 2


# --------------------------------------------------------------------------- #
# Engine: EOS handling
# --------------------------------------------------------------------------- #
def test_generate_stops_when_model_emits_eos() -> None:
    """A token id matching backend.capabilities.eos_token_ids halts the loop
    even before --max-tokens, and the yielded step records stop_reason='eos'."""
    backend = FakeBackend(
        tokens={"P": [1]},
        pieces={1: "P", 2: "X", 99: "<|endoftext|>"},
        distributions={
            (1,): [cand(2, "X", 0.9, 0)],
            (1, 2): [cand(99, "<|endoftext|>", 0.95, 0)],
        },
        eos_token_ids=(99,),
    )

    steps = list(generate(backend, "P", Sampler("greedy"), max_tokens=10))

    assert [s.decision.token_id for s in steps] == [2, 99]
    assert steps[-1].stop_reason == "eos"


def test_generate_continues_past_eos_when_respect_eos_false() -> None:
    """Probing mode: ignore EOS and keep going up to max_tokens."""
    backend = FakeBackend(
        tokens={"P": [1]},
        pieces={1: "P", 99: "<|endoftext|>", 7: "Y"},
        distributions={
            (1,): [cand(99, "<|endoftext|>", 0.95, 0)],
            (1, 99): [cand(7, "Y", 0.9, 0)],
        },
        eos_token_ids=(99,),
    )

    steps = list(
        generate(
            backend,
            "P",
            Sampler("greedy"),
            max_tokens=2,
            respect_eos=False,
        )
    )

    assert [s.decision.token_id for s in steps] == [99, 7]
    assert steps[-1].stop_reason == "max_tokens"


def test_generate_records_user_stop_reason() -> None:
    """When the chosen id matches a --stop id we record stop_reason='user_stop'."""
    backend = FakeBackend(
        tokens={"P": [1]},
        pieces={1: "P", 2: "X"},
        distributions={(1,): [cand(2, "X", 0.9, 0)]},
    )

    steps = list(generate(backend, "P", Sampler("greedy"), max_tokens=5, stop_ids=[2]))

    assert len(steps) == 1
    assert steps[0].stop_reason == "user_stop"


def test_generate_propagates_watch_ids_into_step_watched() -> None:
    """The unified Decode workbench surfaces watch columns on every
    generation step (not just inspect). The engine has to thread
    ``watch_ids`` through to each ``next_distribution`` call so each
    emitted ``GenStep.step_result.watched`` carries an entry per
    requested id. With a backend that puts the watch id in the top-k,
    the entry's rank should match what ``next_distribution`` reports.
    """
    backend = FakeBackend(
        tokens={"P": [1]},
        pieces={1: "P", 2: "X", 7: "W"},
        distributions={
            (1,): [cand(2, "X", 0.7, 0), cand(7, "W", 0.2, 1)],
            (1, 2): [cand(2, "X", 0.6, 0), cand(7, "W", 0.3, 1)],
        },
    )

    steps = list(
        generate(backend, "P", Sampler("greedy"), max_tokens=2, watch_ids=[7])
    )

    assert len(steps) == 2
    for gs in steps:
        watched = gs.step_result.watched
        assert 7 in watched, "engine must forward watch_ids to every step"
        assert watched[7].rank == 1
        assert not math.isnan(watched[7].logprob)


def test_generate_appends_prefix_token_ids_after_tokenized_prompt() -> None:
    """Manual mode rides on ``prefix_token_ids``: the engine has to treat
    the input as ``tokenize(prompt) + prefix_token_ids`` so the backend
    sees a single continuous sequence. We verify that the
    ``tokens_before`` field on the first emitted step includes both the
    tokenized prompt AND the prefix; otherwise the per-pick continuation
    would predict from the wrong context.
    """
    backend = FakeBackend(
        tokens={"P": [1]},
        pieces={1: "P", 2: "X", 5: "Y", 9: "Z"},
        distributions={
            (1, 5, 9): [cand(2, "X", 0.9, 0)],
        },
    )

    [gs] = list(
        generate(
            backend,
            "P",
            Sampler("greedy"),
            max_tokens=1,
            prefix_token_ids=[5, 9],
        )
    )
    assert gs.tokens_before == [1, 5, 9]
    assert gs.decision.token_id == 2


def test_generate_records_max_tokens_stop_reason() -> None:
    """When the loop exits via max_tokens (no EOS, no user stop) the LAST step
    is tagged stop_reason='max_tokens'; earlier steps stay None."""
    backend = FakeBackend(
        tokens={"P": [1]},
        pieces={1: "P", 2: "X", 3: "Y"},
        distributions={
            (1,): [cand(2, "X", 0.9, 0)],
            (1, 2): [cand(3, "Y", 0.9, 0)],
        },
    )

    steps = list(generate(backend, "P", Sampler("greedy"), max_tokens=2))

    assert [s.stop_reason for s in steps] == [None, "max_tokens"]


# --------------------------------------------------------------------------- #
# Samplers (built-ins + custom plug-in)
# --------------------------------------------------------------------------- #
def _ctx() -> SamplerContext:
    return SamplerContext(step=0, token_ids=[], rng=random.Random(0))


def test_sampler_top_p_keeps_at_least_first_candidate() -> None:
    sampler = Sampler("top_p", top_p=0.0)
    cands = [cand(1, "A", 0.7, 0), cand(2, "B", 0.2, 1), cand(3, "C", 0.1, 2)]

    decision = sampler(cands, _ctx())

    assert decision.token_id == 1
    assert [(c.token_id, p) for c, p in decision.kept] == [(1, 1.0)]


def test_sampler_top_p_keeps_cumulative_mass() -> None:
    """top_p=0.85 over {A:0.7, B:0.2, C:0.1} keeps {A, B} (cum=0.9 >= 0.85)."""
    sampler = Sampler("top_p", top_p=0.85)
    cands = [cand(1, "A", 0.7, 0), cand(2, "B", 0.2, 1), cand(3, "C", 0.1, 2)]

    decision = sampler(cands, _ctx())

    assert sorted(c.token_id for c, _ in decision.kept) == [1, 2]


def test_sampler_top_k_truncates_after_temperature() -> None:
    sampler = Sampler("top_k", top_k=2, temperature=1.0)
    cands = [cand(1, "A", 0.5, 0), cand(2, "B", 0.3, 1), cand(3, "C", 0.2, 2)]

    decision = sampler(cands, _ctx())

    assert len(decision.kept) == 2
    assert decision.greedy_token_id == 1


def test_sampler_min_p_drops_candidates_below_relative_threshold() -> None:
    sampler = Sampler("min_p", min_p=0.5)
    # After softmax (T=1) of the given logprobs the order/ratios are preserved.
    cands = [cand(1, "A", 0.6, 0), cand(2, "B", 0.39, 1), cand(3, "C", 0.01, 2)]

    decision = sampler(cands, _ctx())

    kept_ids = [c.token_id for c, _ in decision.kept]
    assert 1 in kept_ids
    assert 3 not in kept_ids  # too far below top-1


def test_sampler_min_p_always_keeps_at_least_top_when_threshold_excludes_all() -> None:
    sampler = Sampler("min_p", min_p=1.01)  # impossible threshold
    cands = [cand(1, "A", 0.6, 0), cand(2, "B", 0.3, 1)]

    decision = sampler(cands, _ctx())

    assert len(decision.kept) == 1
    assert decision.kept[0][0].token_id == 1


def test_sampler_typical_keeps_at_least_one_and_respects_mass() -> None:
    sampler = Sampler("typical", typical_p=0.5)
    cands = [cand(1, "A", 0.5, 0), cand(2, "B", 0.3, 1), cand(3, "C", 0.2, 2)]

    decision = sampler(cands, _ctx())

    assert len(decision.kept) >= 1
    assert all(c.token_id in (1, 2, 3) for c, _ in decision.kept)


def test_sampler_temperature_zero_is_greedy_regardless_of_distribution() -> None:
    sampler = Sampler("greedy", temperature=0.0)
    cands = [cand(1, "A", 0.34, 0), cand(2, "B", 0.33, 1), cand(3, "C", 0.33, 2)]

    decision = sampler(cands, _ctx())

    assert decision.token_id == 1
    assert decision.note == "greedy (argmax)"
    assert decision.changed_greedy is False


def test_sampler_high_temperature_flattens_distribution_into_runner_up() -> None:
    """With a very high temperature and a fixed seed, sampling can pick non-greedy.

    We check the post-temperature renormalized probs in ``kept`` to confirm the
    distribution was actually flattened (top-1 prob drops noticeably below 1).
    """
    sampler = Sampler("temperature", temperature=100.0)
    cands = [cand(1, "A", 0.6, 0), cand(2, "B", 0.39, 1), cand(3, "C", 0.01, 2)]

    decision = sampler(cands, _ctx())

    top_renorm = next(p for c, p in decision.kept if c.token_id == 1)
    assert top_renorm < 0.6  # flatter than the raw distribution


def test_sampler_empty_candidates_raises() -> None:
    with pytest.raises(ValueError, match="no candidates"):
        Sampler("greedy")([], _ctx())


# --------------------------------------------------------------------------- #
# Penalties (repetition / frequency / presence) — applied to logprobs before
# the softmax, using the running history from ``SamplerContext``.
# --------------------------------------------------------------------------- #
def _ctx_with_history(ids: list[int]) -> SamplerContext:
    return SamplerContext(step=len(ids), token_ids=list(ids), rng=random.Random(0))


def test_sampler_repetition_penalty_demotes_seen_tokens() -> None:
    """A repeated token's renormalized prob must drop vs the no-penalty run.

    Token 1 was emitted; with ``repetition_penalty=1.5`` its logprob is
    divided/multiplied (depending on sign) by 1.5, which after softmax
    leaves its renorm-prob smaller than the bare run.
    """
    cands = [cand(1, "A", 0.6, 0), cand(2, "B", 0.4, 1)]
    bare = Sampler("temperature", temperature=1.0).decide(cands, _ctx_with_history([1, 1, 1]))
    pen = Sampler(
        "temperature", temperature=1.0, repetition_penalty=1.5
    ).decide(cands, _ctx_with_history([1, 1, 1]))
    bare_p = next(p for c, p in bare.kept if c.token_id == 1)
    pen_p = next(p for c, p in pen.kept if c.token_id == 1)
    assert pen_p < bare_p
    assert "rep=1.5" in pen.note


def test_sampler_frequency_penalty_scales_with_count() -> None:
    """``frequency_penalty`` subtracts ``freq * count`` from each token's logprob.

    Token 1 appears 4 times; token 2 once. Net effect: token 1's logprob
    drops by ``4*freq``, token 2's by ``freq``. After softmax we expect
    token 1's prob to fall much further than the baseline.
    """
    cands = [cand(1, "A", 0.55, 0), cand(2, "B", 0.45, 1)]
    bare = Sampler("temperature", temperature=1.0).decide(cands, _ctx_with_history([1, 1, 1, 1, 2]))
    pen = Sampler(
        "temperature", temperature=1.0, frequency_penalty=0.5
    ).decide(cands, _ctx_with_history([1, 1, 1, 1, 2]))
    bare_p = next(p for c, p in bare.kept if c.token_id == 1)
    pen_p = next(p for c, p in pen.kept if c.token_id == 1)
    assert pen_p < bare_p
    assert "freq=0.5" in pen.note


def test_sampler_presence_penalty_flat_per_unique_token() -> None:
    """``presence_penalty`` subtracts a fixed value once per *seen* token.

    Two seen tokens get the same -1.0 shift regardless of how many
    times they appeared in history; an unseen third token is left
    untouched. So an unseen runner-up gets relatively boosted.
    """
    cands = [cand(1, "A", 0.5, 0), cand(2, "B", 0.3, 1), cand(3, "C", 0.2, 2)]
    bare = Sampler("temperature", temperature=1.0).decide(cands, _ctx_with_history([1, 1, 2]))
    pen = Sampler(
        "temperature", temperature=1.0, presence_penalty=1.0
    ).decide(cands, _ctx_with_history([1, 1, 2]))
    bare_p3 = next(p for c, p in bare.kept if c.token_id == 3)
    pen_p3 = next(p for c, p in pen.kept if c.token_id == 3)
    # Token 3 (never seen) gets a boost because everybody else got demoted.
    assert pen_p3 > bare_p3
    assert "pres=1" in pen.note


def test_sampler_default_penalties_are_noops() -> None:
    """Defaults (rep=1.0, freq=0, pres=0) must produce bit-identical decisions.

    Otherwise every legacy run silently changes behaviour the moment we
    add penalty fields to ``Sampler``. The history is non-empty on
    purpose -- a no-op penalty against a non-empty history is the
    same shape as the historical pre-penalty Sampler.
    """
    cands = [cand(1, "A", 0.55, 0), cand(2, "B", 0.45, 1)]
    old_style = Sampler("temperature", temperature=1.0).decide(
        cands, _ctx_with_history([1, 2, 1])
    )
    same = Sampler(
        "temperature",
        temperature=1.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ).decide(cands, _ctx_with_history([1, 2, 1]))
    assert old_style.token_id == same.token_id
    assert [p for _, p in old_style.kept] == pytest.approx([p for _, p in same.kept])
    assert "rep" not in same.note
    assert "freq" not in same.note
    assert "pres" not in same.note


# --------------------------------------------------------------------------- #
# Mirostat v2 (local implementation; cloud forwarding tested in test_backends_http)
# --------------------------------------------------------------------------- #
def test_mirostat_initializes_mu_to_2_tau() -> None:
    """First call sets ``_mirostat_mu`` close to ``2 * τ`` per the paper bootstrap.

    The exact value after the first decision is slightly off because the
    per-step μ update fires immediately (μ -= η * (surprise - τ)). What
    we assert here is the *order of magnitude*: after one step μ is
    still in the ballpark of 2τ, never zero or wildly negative.
    """
    sampler = Sampler("mirostat", temperature=1.0, mirostat_target=3.0, mirostat_lr=0.1)
    cands = [cand(1, "A", 0.9, 0), cand(2, "B", 0.1, 1)]
    sampler.decide(cands, _ctx())
    assert sampler._mirostat_mu is not None
    assert sampler._mirostat_mu == pytest.approx(2.0 * 3.0, rel=0.5)
    # The note must reveal the running μ so the educational UI can show
    # how mirostat is converging.
    decision = sampler.decide(cands, _ctx())
    assert "mirostat" in decision.note
    assert "μ=" in decision.note


def test_mirostat_filter_caps_candidates_by_surprise() -> None:
    """Mirostat keeps only candidates with -log(p) <= μ; never empty."""
    # A tight τ forces μ small immediately; the long-tail candidates
    # have high surprise (-log p large) and must be filtered out.
    sampler = Sampler("mirostat", temperature=1.0, mirostat_target=0.2, mirostat_lr=0.5)
    cands = [cand(1, "A", 0.94, 0), cand(2, "B", 0.05, 1), cand(3, "C", 0.01, 2)]
    # Warm up so μ shrinks.
    for _ in range(5):
        sampler.decide(cands, _ctx())
    decision = sampler.decide(cands, _ctx())
    # Surprise(A) ≈ 0.062 nats; B ≈ 3.0; C ≈ 4.6. Expect B and C dropped.
    assert {c.token_id for c, _ in decision.kept} == {1}


def test_mirostat_fallback_returns_top_candidate_when_filter_empties_set() -> None:
    """μ may shrink below the smallest surprise; we still return SOMETHING."""
    sampler = Sampler("mirostat", temperature=1.0, mirostat_target=0.01, mirostat_lr=2.0)
    cands = [cand(1, "A", 0.55, 0), cand(2, "B", 0.45, 1)]
    # Tight τ + aggressive η + few iterations -> μ << any surprise.
    for _ in range(10):
        sampler.decide(cands, _ctx())
    decision = sampler.decide(cands, _ctx())
    # Must yield exactly one kept entry (the most likely token).
    assert len(decision.kept) == 1


def test_mirostat_builder_creates_correct_sampler() -> None:
    """make_sampler('mirostat', ...) carries through to Sampler fields."""
    from decoding_sandbox.core.samplers import make_sampler

    s = make_sampler("mirostat", mirostat_target=4.0, mirostat_lr=0.25, temperature=0.8)
    assert s.name == "mirostat"
    assert s.mirostat_target == pytest.approx(4.0)
    assert s.mirostat_lr == pytest.approx(0.25)
    assert s.temperature == pytest.approx(0.8)


def test_make_sampler_unknown_raises_keyerror() -> None:
    from decoding_sandbox.core.samplers import make_sampler

    with pytest.raises(KeyError, match="Unknown sampler"):
        make_sampler("does-not-exist")


def test_make_sampler_passes_through_parameters() -> None:
    from decoding_sandbox.core.samplers import make_sampler

    s = make_sampler("top_p", top_p=0.5, temperature=0.7)
    assert s.top_p == 0.5
    assert s.temperature == 0.7


def test_load_custom_with_int_return_wraps_decision(tmp_path) -> None:
    plug = tmp_path / "custom.py"
    plug.write_text("def decode(cands, ctx):\n    return cands[-1].token_id\n")

    from decoding_sandbox.core.samplers import load_custom

    fn = load_custom(str(plug))
    cands = [cand(1, "A", 0.7, 0), cand(2, "B", 0.3, 1)]
    decision = fn(cands, _ctx())

    assert decision.token_id == 2
    assert decision.token_text == "B"
    assert decision.greedy_token_id == 1
    assert decision.changed_greedy is True
    assert decision.note == "custom:decode"


def test_load_custom_with_decision_return_is_passed_through(tmp_path) -> None:
    plug = tmp_path / "custom.py"
    plug.write_text(
        "from decoding_sandbox.core.samplers import SamplerDecision\n"
        "def my_decode(cands, ctx):\n"
        "    return SamplerDecision(\n"
        "        token_id=cands[0].token_id,\n"
        "        token_text=cands[0].text,\n"
        "        kept=[(cands[0], 1.0)],\n"
        "        greedy_token_id=cands[0].token_id,\n"
        "        note='inline',\n"
        "    )\n"
    )

    from decoding_sandbox.core.samplers import load_custom

    fn = load_custom(f"{plug}:my_decode")
    cands = [cand(1, "A", 0.7, 0)]
    decision = fn(cands, _ctx())

    assert decision.note == "inline"
    assert decision.token_id == 1


def test_load_custom_missing_function_raises(tmp_path) -> None:
    plug = tmp_path / "custom.py"
    plug.write_text("def other(cands, ctx): return cands[0].token_id\n")

    from decoding_sandbox.core.samplers import load_custom

    with pytest.raises(AttributeError):
        load_custom(str(plug))  # default :decode does not exist


def test_load_custom_unloadable_path_raises() -> None:
    from decoding_sandbox.core.samplers import load_custom

    with pytest.raises((ImportError, FileNotFoundError)):
        load_custom("/this/path/definitely/does/not/exist.py")
