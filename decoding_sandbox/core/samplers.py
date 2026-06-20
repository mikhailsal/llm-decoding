"""Decoding/sampling functions and a plug-in registry.

Samplers operate on a ranked list of ``TokenCandidate`` (each carries a
``logprob``), so the *same* sampler works for a full-vocab HF distribution and a
top-k llama.cpp/cloud distribution -- just request a large ``top_k`` so the
sampler has enough candidates to work with.

A sampler returns a ``SamplerDecision`` (chosen token + the set it kept after
filtering + what greedy would have done), which the ``generate`` UI uses to show
exactly how the sampler changed the outcome.

Write your own:
    def decode(cands, ctx):           # cands: list[TokenCandidate], ctx: SamplerContext
        return cands[0].token_id      # or return a SamplerDecision
Drop it in a .py file and run:  dsbx generate ... --sampler custom --custom-file mine.py:decode
"""

from __future__ import annotations

import importlib.util
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from decoding_sandbox.core.types import TokenCandidate


@dataclass
class SamplerContext:
    step: int
    token_ids: list[int]
    rng: random.Random


@dataclass
class SamplerDecision:
    token_id: int
    token_text: str
    kept: list[tuple[TokenCandidate, float]] = field(default_factory=list)  # (cand, renorm prob)
    greedy_token_id: int | None = None
    note: str = ""

    @property
    def changed_greedy(self) -> bool:
        return self.greedy_token_id is not None and self.token_id != self.greedy_token_id


def _softmax(logprobs: list[float]) -> list[float]:
    m = max(logprobs)
    exps = [math.exp(lp - m) for lp in logprobs]
    s = sum(exps)
    return [e / s for e in exps]


@dataclass
class Sampler:
    """A configurable sampler: temperature, then optional truncation filters."""

    name: str
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    typical_p: float | None = None

    def __call__(self, cands: Sequence[TokenCandidate], ctx: SamplerContext) -> SamplerDecision:
        return self.decide(cands, ctx)

    def decide(self, cands: Sequence[TokenCandidate], ctx: SamplerContext) -> SamplerDecision:
        if not cands:
            raise ValueError("no candidates to sample from")
        greedy_id = cands[0].token_id

        if self.temperature is not None and self.temperature <= 0:
            top = cands[0]
            return SamplerDecision(
                token_id=top.token_id,
                token_text=top.text,
                kept=[(top, 1.0)],
                greedy_token_id=greedy_id,
                note="greedy (argmax)",
            )

        temp = self.temperature or 1.0
        probs = _softmax([c.logprob / temp for c in cands])
        pairs: list[tuple[TokenCandidate, float]] = sorted(
            zip(cands, probs), key=lambda x: x[1], reverse=True
        )

        notes: list[str] = [f"T={temp:g}"]
        if self.top_k:
            pairs = pairs[: self.top_k]
            notes.append(f"top_k={self.top_k}")
        if self.top_p is not None:
            pairs = _filter_top_p(pairs, self.top_p)
            notes.append(f"top_p={self.top_p:g}")
        if self.min_p is not None:
            pairs = _filter_min_p(pairs, self.min_p)
            notes.append(f"min_p={self.min_p:g}")
        if self.typical_p is not None:
            pairs = _filter_typical(pairs, self.typical_p)
            notes.append(f"typical_p={self.typical_p:g}")

        kept_cands = [c for c, _ in pairs]
        kept_w = [p for _, p in pairs]
        total = sum(kept_w) or 1.0
        kept_norm = [w / total for w in kept_w]
        chosen = ctx.rng.choices(kept_cands, weights=kept_norm, k=1)[0]
        return SamplerDecision(
            token_id=chosen.token_id,
            token_text=chosen.text,
            kept=list(zip(kept_cands, kept_norm)),
            greedy_token_id=greedy_id,
            note=", ".join(notes),
        )


def _filter_top_p(pairs: list[tuple[TokenCandidate, float]], p: float):
    out, cum = [], 0.0
    for c, w in pairs:
        out.append((c, w))
        cum += w
        if cum >= p:
            break
    return out


def _filter_min_p(pairs: list[tuple[TokenCandidate, float]], min_p: float):
    if not pairs:
        return pairs
    thresh = min_p * pairs[0][1]
    return [(c, w) for c, w in pairs if w >= thresh] or [pairs[0]]


def _filter_typical(pairs: list[tuple[TokenCandidate, float]], mass: float):
    if not pairs:
        return pairs
    ent = -sum(w * math.log(w) for _, w in pairs if w > 0)
    scored = sorted(pairs, key=lambda cw: abs(-math.log(cw[1]) - ent) if cw[1] > 0 else 1e9)
    out, cum = [], 0.0
    for c, w in scored:
        out.append((c, w))
        cum += w
        if cum >= mass:
            break
    return out


# --------------------------------------------------------------------------- #
# Registry of built-in samplers (name -> builder taking kwargs).
# --------------------------------------------------------------------------- #
def _greedy(**_):
    return Sampler("greedy", temperature=0.0)


def _temperature(temperature: float = 0.8, **_):
    return Sampler("temperature", temperature=temperature)


def _top_k(top_k: int = 40, temperature: float = 1.0, **_):
    return Sampler("top_k", temperature=temperature, top_k=top_k)


def _top_p(top_p: float = 0.9, temperature: float = 1.0, **_):
    return Sampler("top_p", temperature=temperature, top_p=top_p)


def _min_p(min_p: float = 0.05, temperature: float = 1.0, **_):
    return Sampler("min_p", temperature=temperature, min_p=min_p)


def _typical(typical_p: float = 0.95, temperature: float = 1.0, **_):
    return Sampler("typical", temperature=temperature, typical_p=typical_p)


BUILTINS: dict[str, Callable[..., Sampler]] = {
    "greedy": _greedy,
    "temperature": _temperature,
    "top_k": _top_k,
    "top_p": _top_p,
    "min_p": _min_p,
    "typical": _typical,
}


def make_sampler(name: str, **params) -> Sampler:
    if name not in BUILTINS:
        raise KeyError(f"Unknown sampler '{name}'. Available: {sorted(BUILTINS)}")
    return BUILTINS[name](**params)


CustomFn = Callable[[Sequence[TokenCandidate], SamplerContext], "int | SamplerDecision"]


def load_custom(spec: str) -> Callable[[Sequence[TokenCandidate], SamplerContext], SamplerDecision]:
    """Load a custom sampler from 'path/to/file.py:func' (func defaults to 'decode').

    The function may return either a token_id (int) or a SamplerDecision.
    """
    path, _, func_name = spec.partition(":")
    func_name = func_name or "decode"
    smod = importlib.util.spec_from_file_location("dsbx_custom_sampler", path)
    if smod is None or smod.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(smod)
    smod.loader.exec_module(module)
    fn: CustomFn = getattr(module, func_name)

    def wrapped(cands: Sequence[TokenCandidate], ctx: SamplerContext) -> SamplerDecision:
        result = fn(cands, ctx)
        if isinstance(result, SamplerDecision):
            return result
        token_id = int(result)
        text = next((c.text for c in cands if c.token_id == token_id), "")
        return SamplerDecision(
            token_id=token_id,
            token_text=text,
            kept=[(c, c.prob) for c in cands],
            greedy_token_id=cands[0].token_id if cands else None,
            note=f"custom:{func_name}",
        )

    return wrapped
