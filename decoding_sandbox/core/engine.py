"""Decoding loop shared by the CLI (and the future web UI).

Pulls a next-token distribution from the backend, lets a sampler pick a token,
appends it, and repeats -- yielding a ``GenStep`` per step so the caller can
visualize what happened without re-implementing the loop.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.samplers import SamplerContext, SamplerDecision
from decoding_sandbox.core.types import StepResult, TokenCandidate


@dataclass
class GenStep:
    step: int
    tokens_before: list[int]
    step_result: StepResult
    decision: SamplerDecision

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
) -> Iterator[GenStep]:
    rng = rng or random.Random()
    tokens = backend.tokenize(prompt)
    for s in range(max_tokens):
        sr = backend.next_distribution(tokens, top_k)
        if not sr.candidates:
            break
        ctx = SamplerContext(step=s, token_ids=list(tokens), rng=rng)
        decision: SamplerDecision = sampler(sr.candidates, ctx)
        yield GenStep(step=s, tokens_before=list(tokens), step_result=sr, decision=decision)
        tokens.append(decision.token_id)
        if decision.token_id in stop_ids:
            break
