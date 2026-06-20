"""Core data model shared by every backend and UI.

These types are deliberately backend-agnostic: a full-vocab HF forward pass, a
top-k llama.cpp response, and a cloud provider's top_logprobs all reduce to the
same ``StepResult`` so the inspect/generate/manual UIs never special-case a
backend -- they only read ``Capabilities`` to decide what to show.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TokenCandidate:
    """One candidate token in a distribution at a single position.

    ``is_special`` is set by backends that can tell (HF via
    ``tokenizer.all_special_ids``, llama-cpp-py via ``Llama.token_eos()`` +
    the ``<|...|>`` heuristic). The renderer uses it to colour the token
    distinctively so the user can immediately see EOS/BOS/PAD without
    eyeballing strings like ``<|endoftext|>``. Backends that don't expose
    this info leave it ``False`` -- the renderer falls back to a pattern
    check on the text.
    """

    token_id: int
    text: str
    logprob: float
    rank: int  # 0 = most likely
    is_special: bool = False

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)


@dataclass
class StepResult:
    """The model's predicted distribution at one position.

    Used both for *inspection* (``chosen`` = the actual next token already in the
    text, so we can show the probability the model assigned to reality) and for
    *generation* (``chosen`` = the token the sampler picked).
    """

    position: int
    candidates: list[TokenCandidate]  # ranked, most likely first
    is_full_vocab: bool
    chosen: TokenCandidate | None = None
    # The token text at this position (the context token being conditioned on),
    # handy for rendering inspect rows. Optional.
    context_text: str | None = None
    # Probability of specific "watch" tokens at this position, even if they fall
    # outside the top-k. A candidate with rank == -1 / nan logprob means unknown
    # (token outside a non-full-vocab backend's returned top-k).
    watched: dict[int, TokenCandidate] = field(default_factory=dict)

    @property
    def top(self) -> TokenCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def confidence(self) -> float:
        """Max probability (top-1) -- the model's confidence at this position."""
        t = self.top
        return t.prob if t else 0.0

    def find(self, token_id: int) -> TokenCandidate | None:
        for c in self.candidates:
            if c.token_id == token_id:
                return c
        return None


@dataclass
class Capabilities:
    """What a backend can do, so the UI can adapt instead of guessing.

    ``eos_token_ids`` lists every token id the backend believes terminates a
    generation. Set non-empty by backends that expose it (HF reads
    ``model.config.eos_token_id``, llama-cpp-py reads ``Llama.token_eos()``).
    The ``generate`` engine treats any chosen token id in this set as an
    implicit stop, so a base model that wants to emit ``<|endoftext|>``
    actually halts instead of running until ``--max-tokens``.
    """

    name: str
    full_vocab: bool  # exact distribution over the entire vocabulary
    prompt_logprobs: bool  # can score every prompt token (whole context)
    max_top_logprobs: int  # how many candidates per position it can return
    can_force_token: bool = False  # supports manual/forced token decoding
    notes: str = ""
    eos_token_ids: tuple[int, ...] = ()


__all__ = ["TokenCandidate", "StepResult", "Capabilities", "field"]
