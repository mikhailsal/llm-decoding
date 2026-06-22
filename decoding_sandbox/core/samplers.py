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
from collections import Counter
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
    """A configurable sampler: temperature, then optional truncation filters.

    Penalties (``repetition_penalty`` / ``frequency_penalty`` /
    ``presence_penalty``) are applied to the raw logprobs *before* the
    temperature softmax, using the running token history from
    :class:`SamplerContext`. Defaults of ``1.0`` / ``0.0`` are no-ops so
    samplers built without them stay bit-identical to the historical
    behaviour. The cloud path forwards the same values as plain body
    fields when ``provider.supports_repetition_penalty`` (etc.) is on.

    ``mirostat_target`` opts in to Mirostat v2: a perplexity-targeting
    loop that dynamically truncates candidates so the per-step surprise
    converges to ``mirostat_target`` (in nats, NOT bits -- we use nats
    so it composes cleanly with logprobs). State (``_mirostat_mu``) is
    kept on the instance so one Sampler ``Sampler`` object IS one
    generation -- which is exactly how the engine uses them today (one
    ``make_sampler`` call per ``generate`` invocation).
    """

    name: str
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    typical_p: float | None = None
    # Penalties. Defaults are explicit no-ops to make "did we opt in"
    # easy to read at call sites and at sampler_to_api_params time.
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # Mirostat v2 knobs. Set ``mirostat_target`` (tau, in nats) to a
    # positive number to enable. ``mirostat_lr`` is the EMA learning
    # rate (eta in the paper). Both ignored when target is None.
    mirostat_target: float | None = None
    mirostat_lr: float = 0.1
    # Internal mirostat state. Initialized lazily to ``2 * tau`` on the
    # first call (the paper's recommended bootstrap value). Not a
    # dataclass field so ``Sampler`` stays cheaply hashable / printable.
    _mirostat_mu: float | None = field(default=None, repr=False)

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
        notes: list[str] = [f"T={temp:g}"]
        # Apply penalties to a working copy of each logprob. We never
        # mutate the input ``TokenCandidate``s -- they belong to the
        # caller (a Backend or the inspector UI), and the manual TUI
        # in particular re-uses them across renders.
        adjusted: list[float] = []
        seen_counts = Counter(ctx.token_ids) if (
            self.repetition_penalty != 1.0
            or self.frequency_penalty != 0.0
            or self.presence_penalty != 0.0
        ) else None
        if seen_counts is not None:
            for c in cands:
                lp = c.logprob
                count = seen_counts.get(c.token_id, 0)
                if count and self.repetition_penalty != 1.0:
                    # llama.cpp convention: positive logprobs divide by the
                    # penalty (decrease, since penalty >= 1.0), negative
                    # logprobs multiply (also decrease). Symmetric for
                    # ``penalty < 1.0`` to encourage repetition.
                    if lp > 0:
                        lp /= self.repetition_penalty
                    else:
                        lp *= self.repetition_penalty
                if count and self.frequency_penalty != 0.0:
                    lp -= self.frequency_penalty * count
                if count and self.presence_penalty != 0.0:
                    lp -= self.presence_penalty
                adjusted.append(lp)
            if self.repetition_penalty != 1.0:
                notes.append(f"rep={self.repetition_penalty:g}")
            if self.frequency_penalty != 0.0:
                notes.append(f"freq={self.frequency_penalty:g}")
            if self.presence_penalty != 0.0:
                notes.append(f"pres={self.presence_penalty:g}")
        else:
            adjusted = [c.logprob for c in cands]
        probs = _softmax([lp / temp for lp in adjusted])
        pairs: list[tuple[TokenCandidate, float]] = sorted(
            zip(cands, probs), key=lambda x: x[1], reverse=True
        )

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
        if self.mirostat_target is not None and self.mirostat_target > 0:
            pairs = self._filter_mirostat(pairs)
            notes.append(
                f"mirostat(τ={self.mirostat_target:g},η={self.mirostat_lr:g},μ={self._mirostat_mu:.2f})"
            )

        kept_cands = [c for c, _ in pairs]
        kept_w = [p for _, p in pairs]
        total = sum(kept_w) or 1.0
        kept_norm = [w / total for w in kept_w]
        chosen = ctx.rng.choices(kept_cands, weights=kept_norm, k=1)[0]
        # Mirostat post-update: shift μ towards the surprise the chosen
        # token actually emitted (in nats, since pairs carry probs).
        if self.mirostat_target is not None and self.mirostat_target > 0:
            chosen_prob = next(
                (p for c, p in zip(kept_cands, kept_norm) if c.token_id == chosen.token_id),
                None,
            )
            if chosen_prob and chosen_prob > 0:
                surprise = -math.log(chosen_prob)
                self._mirostat_mu -= self.mirostat_lr * (surprise - self.mirostat_target)
        return SamplerDecision(
            token_id=chosen.token_id,
            token_text=chosen.text,
            kept=list(zip(kept_cands, kept_norm)),
            greedy_token_id=greedy_id,
            note=", ".join(notes),
        )

    def _filter_mirostat(
        self, pairs: list[tuple[TokenCandidate, float]]
    ) -> list[tuple[TokenCandidate, float]]:
        """Mirostat v2: keep candidates with surprise <= μ; lazy-init μ.

        At the very first step we don't have a running μ estimate, so
        we use the paper's recommended ``2 * τ`` bootstrap. The
        per-step μ-update happens in the caller AFTER a token is
        sampled so the update reflects the realised surprise.
        """
        if self._mirostat_mu is None:
            self._mirostat_mu = 2.0 * float(self.mirostat_target or 0.0)
        cap = self._mirostat_mu
        kept = [(c, p) for c, p in pairs if p > 0 and (-math.log(p)) <= cap]
        # μ shrinks below the smallest available surprise on rare-token
        # runs; fall back to the single most likely candidate so the
        # sampler never returns an empty kept-set (which would crash
        # ``rng.choices``).
        return kept or pairs[:1]


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
def _penalty_kwargs(params: dict) -> dict:
    """Extract penalty knobs from ``params`` with no-op defaults.

    All builders accept the same penalty kwargs so the UI can keep its
    sampler-agnostic penalty inputs. Defaults match
    :class:`Sampler` (``1.0`` / ``0.0``) so callers that don't pass
    anything stay bit-identical to the historical behaviour.
    """
    return {
        "repetition_penalty": float(params.get("repetition_penalty", 1.0)),
        "frequency_penalty": float(params.get("frequency_penalty", 0.0)),
        "presence_penalty": float(params.get("presence_penalty", 0.0)),
    }


def _greedy(**params):
    return Sampler("greedy", temperature=0.0, **_penalty_kwargs(params))


def _temperature(temperature: float = 0.8, **params):
    return Sampler("temperature", temperature=temperature, **_penalty_kwargs(params))


def _top_k(top_k: int = 40, temperature: float = 1.0, **params):
    return Sampler("top_k", temperature=temperature, top_k=top_k, **_penalty_kwargs(params))


def _top_p(top_p: float = 0.9, temperature: float = 1.0, **params):
    return Sampler("top_p", temperature=temperature, top_p=top_p, **_penalty_kwargs(params))


def _min_p(min_p: float = 0.05, temperature: float = 1.0, **params):
    return Sampler("min_p", temperature=temperature, min_p=min_p, **_penalty_kwargs(params))


def _typical(typical_p: float = 0.95, temperature: float = 1.0, **params):
    return Sampler(
        "typical", temperature=temperature, typical_p=typical_p, **_penalty_kwargs(params)
    )


def _mirostat(
    mirostat_target: float = 5.0,
    mirostat_lr: float = 0.1,
    temperature: float = 1.0,
    **params,
):
    """Mirostat v2 builder.

    ``mirostat_target`` is the per-step target surprise in nats (so
    e.g. ``5.0`` ≈ 7.2 bits ≈ the typical perplexity Llama-2 chat
    targets). The default lr matches the paper's recommended ``0.1``.
    Passes through the standard penalty kwargs so users can stack
    mirostat with a small ``repetition_penalty`` like llama.cpp does.
    """
    return Sampler(
        "mirostat",
        temperature=temperature,
        mirostat_target=float(mirostat_target),
        mirostat_lr=float(mirostat_lr),
        **_penalty_kwargs(params),
    )


BUILTINS: dict[str, Callable[..., Sampler]] = {
    "greedy": _greedy,
    "temperature": _temperature,
    "top_k": _top_k,
    "top_p": _top_p,
    "min_p": _min_p,
    "typical": _typical,
    "mirostat": _mirostat,
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
