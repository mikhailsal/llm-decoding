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
