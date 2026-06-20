"""Speculative decoding (draft + target) with per-round accept/reject info.

A cheap *draft* model proposes ``gamma`` tokens greedily; the *target* model
verifies them in a single forward pass and accepts the longest matching prefix,
then emits one correction/bonus token. The output is identical to plain greedy
decoding from the target, but produces multiple tokens per target forward pass.

Draft and target MUST share a tokenizer/vocabulary (e.g. two models from the
same family), since tokens are exchanged as ids. The target backend must expose
``verify_greedy`` (HFBackend does).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

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
        return self.proposed[self.accepted:]


def speculative_generate(
    target: Backend,
    draft: Backend,
    prompt: str,
    *,
    gamma: int = 4,
    max_tokens: int = 24,
) -> Iterator[SpecRound]:
    if not hasattr(target, "verify_greedy"):
        raise TypeError(
            "target backend must support verify_greedy (use the hf backend)."
        )

    tokens = target.tokenize(prompt)
    produced = 0
    step = 0
    while produced < max_tokens:
        # 1) draft proposes gamma tokens greedily
        dctx = list(tokens)
        proposed: list[TokenCandidate] = []
        for _ in range(gamma):
            d = draft.next_distribution(dctx, top_k=1)
            if not d.candidates:
                break
            c = d.candidates[0]
            proposed.append(c)
            dctx.append(c.token_id)

        draft_ids = [c.token_id for c in proposed]
        accepted, correction = target.verify_greedy(tokens, draft_ids)
        emitted = draft_ids[:accepted] + ([correction.token_id] if correction else [])
        tokens.extend(emitted)
        produced += len(emitted)
        yield SpecRound(step, proposed, accepted, correction, emitted)
        step += 1
        if not emitted:
            break
