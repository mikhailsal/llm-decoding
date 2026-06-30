"""Decoding loop shared by the CLI (and the future web UI).

Pulls a next-token distribution from the backend, lets a sampler pick a token,
appends it, and repeats -- yielding a ``GenStep`` per step so the caller can
visualize what happened without re-implementing the loop.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from dsbx.core.backend import Backend
from dsbx.core.samplers import SamplerContext, SamplerDecision
from dsbx.core.types import StepResult, TokenCandidate


@dataclass
class GenStep:
    step: int
    tokens_before: list[int]
    step_result: StepResult
    decision: SamplerDecision
    # Filled in when the loop terminates *because of* this step. Lets callers
    # render a "stopped on EOS" footer without re-checking the model.
    stop_reason: str | None = None  # None | "eos" | "user_stop" | "max_tokens"

    def chosen_candidate(self) -> TokenCandidate | None:
        return self.step_result.find(self.decision.token_id)


def generate(
    backend: Backend,
    prompt: str,
    sampler,
    *,
    max_tokens: int = 20,
    top_k: int = 50,
    rng: random.Random | None = None,
    stop_ids: Sequence[int] = (),
    respect_eos: bool = True,
    watch_ids: Sequence[int] = (),
    prefix_token_ids: Sequence[int] = (),
) -> Iterator[GenStep]:
    """Decode tokens until ``max_tokens``, a ``--stop`` id, or EOS.

    The model's EOS ids come from ``backend.capabilities.eos_token_ids``
    (HF reads ``model.config.eos_token_id``; llama-cpp-py reads
    ``Llama.token_eos()``). Backends that don't expose them get the
    historical "run to ``max_tokens``" behaviour. Set ``respect_eos=False``
    to inspect what the model would emit *past* EOS (useful when probing
    base-vs-instruct behaviour).

    ``watch_ids`` -- token ids whose per-step probability the caller
    wants to track across every emitted token, even when they fall
    outside the backend's returned top-k. Forwarded straight to
    :meth:`Backend.next_distribution`; full-vocab backends populate
    ``step.watched`` with exact logprobs read from the forward-pass
    tensor, top-k-only backends fall back to "found in top-k or
    rank=-1/NaN". The web layer ties this into the same ``watch_*``
    knobs the inspect path already uses.

    ``prefix_token_ids`` -- token ids appended to the tokenized prompt
    BEFORE the first decode step, so the model sees ``tokenize(prompt) +
    prefix_token_ids`` as one continuous sequence. Powers the unified
    workbench's "manual decoding" mode (the browser holds the user's
    picks and resends the growing list on each pick) without needing
    server-side session state.
    """
    rng = rng or random.Random()
    tokens = backend.tokenize(prompt) + [int(t) for t in prefix_token_ids]
    eos_ids: frozenset[int] = (
        frozenset(backend.capabilities.eos_token_ids) if respect_eos else frozenset()
    )
    user_stop_ids = frozenset(stop_ids)
    watch_id_list = list(watch_ids)
    for s in range(max_tokens):
        sr = backend.next_distribution(tokens, top_k, watch_ids=watch_id_list)
        if not sr.candidates:
            break
        ctx = SamplerContext(step=s, token_ids=list(tokens), rng=rng)
        decision: SamplerDecision = sampler(sr.candidates, ctx)
        stop_reason: str | None = None
        if decision.token_id in eos_ids:
            stop_reason = "eos"
        elif decision.token_id in user_stop_ids:
            stop_reason = "user_stop"
        elif s == max_tokens - 1:
            stop_reason = "max_tokens"
        yield GenStep(
            step=s,
            tokens_before=list(tokens),
            step_result=sr,
            decision=decision,
            stop_reason=stop_reason,
        )
        tokens.append(decision.token_id)
        if stop_reason in ("eos", "user_stop"):
            break
