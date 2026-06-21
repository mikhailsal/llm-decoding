"""Speculative decoding (draft + target) with per-round accept/reject info.

A cheap *draft* model proposes ``gamma`` tokens greedily; the *target* model
verifies them in a single forward pass and accepts the longest matching prefix,
then emits one correction/bonus token. The output is identical to plain greedy
decoding from the target, but produces multiple tokens per target forward pass.

Draft and target MUST share a tokenizer/vocabulary (e.g. two models from the
same family), since tokens are exchanged as ids.

This module defines:
- ``Speculator``  : a Protocol pairing a draft + target with ``propose`` and
                    ``verify`` operations -- the formal interface from the plan.
- ``HFSpeculator``: a concrete implementation that uses any backend that exposes
                    ``verify_greedy`` (HFBackend does) as the target and any
                    backend at all as the draft. New backends (e.g. a future
                    llama.cpp built-in speculative wrapper) just have to satisfy
                    this Protocol.
- ``speculative_generate``: the round-by-round driver. Accepts either a
                            ``Speculator`` or a ``(target, draft)`` pair (for
                            backwards compatibility).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.types import TokenCandidate


@dataclass
class SpecRound:
    step: int
    proposed: list[TokenCandidate]  # what the draft proposed
    accepted: int  # how many were accepted by the target
    correction: TokenCandidate | None  # target's replacement / bonus token
    emitted_ids: list[int] = field(default_factory=list)

    @property
    def rejected(self) -> list[TokenCandidate]:
        return self.proposed[self.accepted :]


@runtime_checkable
class Speculator(Protocol):
    """Pair of (draft, target) backends with propose/verify operations.

    Implementations must share a tokenizer between draft and target (i.e. the
    token ids exchanged by ``propose`` must be meaningful to ``verify``).
    """

    target: Backend
    draft: Backend

    def propose(self, tokens: list[int], gamma: int) -> list[TokenCandidate]:
        """Draft proposes up to ``gamma`` greedy continuations of ``tokens``."""

    def verify(self, tokens: list[int], draft_ids: list[int]) -> tuple[int, TokenCandidate | None]:
        """Target verifies the drafts; returns ``(accepted, correction)``.

        ``accepted`` is how many leading drafts the target's argmax agrees with.
        ``correction`` is either the target's replacement (on the first
        mismatch) or the bonus next token (when all drafts were accepted). May
        be ``None`` only if the target itself has no further continuation.
        """


@dataclass
class HFSpeculator:
    """Concrete Speculator built on any backend with ``verify_greedy``.

    HFBackend supplies the ``verify_greedy`` one-forward-pass implementation;
    the draft can be any Backend that returns greedy continuations via
    ``next_distribution`` (typically a smaller HF model, but a llama.cpp
    backend with a shared tokenizer would also satisfy the Protocol).
    """

    target: Backend
    draft: Backend

    def propose(self, tokens: list[int], gamma: int) -> list[TokenCandidate]:
        proposed: list[TokenCandidate] = []
        ctx = list(tokens)
        for _ in range(gamma):
            d = self.draft.next_distribution(ctx, top_k=1)
            if not d.candidates:
                break
            c = d.candidates[0]
            proposed.append(c)
            ctx.append(c.token_id)
        return proposed

    def verify(self, tokens: list[int], draft_ids: list[int]) -> tuple[int, TokenCandidate | None]:
        verify_greedy = getattr(self.target, "verify_greedy", None)
        if verify_greedy is None:
            raise TypeError(
                "target backend must expose verify_greedy(context_ids, draft_ids) "
                "to act as a speculative-decoding target (HFBackend does)."
            )
        return verify_greedy(tokens, draft_ids)


def speculative_generate(
    target_or_spec: Backend | Speculator,
    draft: Backend | None = None,
    prompt: str = "",
    *,
    gamma: int = 4,
    max_tokens: int = 24,
) -> Iterator[SpecRound]:
    """Drive speculative decoding round by round.

    Call signatures:
        speculative_generate(speculator, prompt=..., gamma=..., max_tokens=...)
        speculative_generate(target, draft, prompt, gamma=..., max_tokens=...)

    The second form is preserved for the existing CLI / tests; internally it
    constructs an :class:`HFSpeculator`.
    """
    if isinstance(target_or_spec, Speculator) and draft is None:
        spec = target_or_spec
    else:
        if draft is None:
            raise TypeError(
                "speculative_generate requires either a Speculator or both "
                "(target, draft) Backends."
            )
        # Default wrapper around verify_greedy-capable target.
        spec = HFSpeculator(target=target_or_spec, draft=draft)

    target = spec.target
    tokens = target.tokenize(prompt)
    produced = 0
    step = 0
    while produced < max_tokens:
        proposed = spec.propose(tokens, gamma)
        draft_ids = [c.token_id for c in proposed]
        accepted, correction = spec.verify(tokens, draft_ids)
        remaining = max_tokens - produced
        emitted = (
            draft_ids[:accepted] + ([correction.token_id] if correction is not None else [])
        )[:remaining]
        tokens.extend(emitted)
        produced += len(emitted)
        yield SpecRound(step, proposed, accepted, correction, emitted)
        step += 1
        if not emitted:
            break


__all__ = ["SpecRound", "Speculator", "HFSpeculator", "speculative_generate"]
