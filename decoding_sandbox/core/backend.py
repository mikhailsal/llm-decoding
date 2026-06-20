"""The Backend protocol every engine implements.

A backend only has to know how to (a) tokenize/detokenize and (b) return the
next-token distribution for a given context. From those, the base class derives
``score_prompt`` (whole-context inspection) generically -- backends that can do
it more efficiently (HF, in one forward pass) override it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate


class Backend(ABC):
    """Abstract decoding backend."""

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def tokenize(self, text: str) -> list[int]: ...

    @abstractmethod
    def detokenize(self, token_ids: list[int]) -> str: ...

    @abstractmethod
    def piece(self, token_id: int) -> str:
        """Human-readable text for a single token id."""

    @abstractmethod
    def next_distribution(self, token_ids: list[int], top_k: int) -> StepResult:
        """Distribution over the token that follows ``token_ids``.

        ``candidates`` must be ranked most-likely first. ``position`` should be
        ``len(token_ids)`` (the index of the predicted token).
        """

    # -- derived ----------------------------------------------------------- #

    def score_prompt(
        self, prompt: str, top_k: int, watch_ids: list[int] | None = None
    ) -> list[StepResult]:
        """Per-position inspection of an existing prompt (whole context).

        Generic implementation: re-evaluate the next-token distribution at each
        prefix and record the probability the model gave to the *actual* next
        token. O(n) backend calls; HF and llamacpp-py override this with a
        single forward pass.

        For an N-token prompt this returns N StepResults. The first N-1 rows
        carry an actual ``chosen`` token (the prompt's real next token). The
        final row -- the distribution conditioned on the full prompt -- has
        ``chosen=None`` and answers "what does the model predict comes
        next?". Watched ids are looked up on this row too, so e.g. P(EOS)
        after the period in "...dry." finally has a place to live.
        """
        ids = self.tokenize(prompt)
        watch_ids = watch_ids or []
        results: list[StepResult] = []
        for i in range(1, len(ids) + 1):
            ctx = ids[:i]
            step = self.next_distribution(ctx, top_k)
            if i < len(ids):
                actual = ids[i]
                chosen = step.find(actual)
                if chosen is None:
                    # Actual token fell outside the returned top-k (only
                    # possible for non-full-vocab backends). Mark its prob as
                    # unknown.
                    chosen = TokenCandidate(
                        token_id=actual,
                        text=self.piece(actual),
                        logprob=math.nan,
                        rank=-1,
                    )
            else:
                # Trailing prediction step: no actual next token to verify
                # against. The renderer reads chosen=None as "?" markers.
                chosen = None
            step.position = i
            step.chosen = chosen
            step.context_text = self.piece(ids[i - 1])
            step.watched = {wid: self.lookup_watch(step, wid) for wid in watch_ids}
            results.append(step)
        return results

    def lookup_watch(self, step: StepResult, token_id: int) -> TokenCandidate:
        """Resolve a watch token's candidate from a step (or mark as <top-k).

        Public so UI code can populate ``step.watched`` after building a
        StepResult outside of the standard ``score_prompt`` loop (e.g. the
        chat-only "next-token" fallback in ``cmd_inspect``).
        """
        found = step.find(token_id)
        if found is not None:
            return found
        return TokenCandidate(
            token_id=token_id, text=self.piece(token_id), logprob=math.nan, rank=-1
        )

    def close(self) -> None:  # optional cleanup
        pass


def candidates_from_logprobs(
    pairs: list[tuple[int, str, float]],
) -> list[TokenCandidate]:
    """Build a ranked candidate list from (token_id, text, logprob) triples."""
    ordered = sorted(pairs, key=lambda p: p[2], reverse=True)
    return [
        TokenCandidate(token_id=tid, text=text, logprob=lp, rank=rank)
        for rank, (tid, text, lp) in enumerate(ordered)
    ]
