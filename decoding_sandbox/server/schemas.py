"""Pydantic wire schemas for the dsbx HTTP server.

The schemas mirror the in-memory dataclasses from
:mod:`decoding_sandbox.core.types` and :mod:`decoding_sandbox.core.samplers`
one-to-one. Keeping them in their own module (only imported on the server
side) means the client can stay pydantic-free: it parses raw JSON dicts into
the existing dataclasses directly.

Three design notes the rest of the protocol depends on:

- ``watched`` is serialized as a list of ``WireWatched(token_id, candidate)``
  entries rather than a ``dict[int, ...]``. JSON object keys are strings; an
  explicit list avoids any client-side ``int(key)`` coercion and keeps the
  schema self-describing.
- ``SamplerDecision.kept`` is a list of ``(TokenCandidate, prob)`` tuples in
  memory; over the wire we flatten it to ``WireKeptEntry(token_id, prob)``
  so the client can rebuild the structure by joining against the step's
  ``candidates`` list (every kept id is guaranteed to appear there).
- ``logprob`` is ``float | None``. A ``None`` on the wire means "the model
  did not return a value for this token" (i.e. it fell outside a
  non-full-vocab backend's top-k); in memory we round-trip this as
  ``math.nan`` so the existing ``watch_cell``/``fmt_prob`` rendering keeps
  working. Plain JSON has no NaN literal -- using None makes the wire
  format strictly standards-compliant and JS-parseable.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Core dataclass mirrors
# --------------------------------------------------------------------------- #


class WireTokenCandidate(BaseModel):
    """Mirror of :class:`decoding_sandbox.core.types.TokenCandidate`.

    ``logprob`` is ``Optional[float]``: ``None`` on the wire stands in
    for ``math.nan`` in memory, since standard JSON has no NaN literal.

    ``sampling_mask_count`` carries the Fireworks NewLogProbs
    ``sampling_mask: 'count'`` value when present. ``None`` means the
    backend either doesn't support the flag or didn't report it for
    this position.
    """

    token_id: int
    text: str
    logprob: float | None = None
    rank: int
    is_special: bool = False
    sampling_mask_count: int | None = None


class WireWatched(BaseModel):
    """One ``watched`` entry on a step: pinned id + its candidate."""

    token_id: int
    candidate: WireTokenCandidate


class WireStepResult(BaseModel):
    """Mirror of :class:`decoding_sandbox.core.types.StepResult`."""

    position: int
    candidates: list[WireTokenCandidate]
    is_full_vocab: bool
    chosen: WireTokenCandidate | None = None
    context_text: str | None = None
    watched: list[WireWatched] = Field(default_factory=list)


class WireCapabilities(BaseModel):
    """Mirror of :class:`decoding_sandbox.core.types.Capabilities`."""

    name: str
    full_vocab: bool
    prompt_logprobs: bool
    max_top_logprobs: int
    can_force_token: bool = False
    notes: str = ""
    eos_token_ids: list[int] = Field(default_factory=list)
    # BOS marker(s) the model uses; empty for models with no canonical
    # BOS (the frontend greys out the "fill BOS" helper in that case).
    bos_token_ids: list[int] = Field(default_factory=list)
    # Provider extensions, see Capabilities docstring.
    supports_ignore_eos: bool = False
    supports_perf_metrics: bool = False
    supports_service_tier: bool = False
    supports_sampling_mask: bool = False
    supports_raw_output: bool = False
    supports_logit_bias: bool = False
    supports_combined_echo_stream: bool = False
    # When true, score_prompt / stream_native accept a non-empty
    # ``prepend_token_ids`` argument; backends that tokenize server-side
    # from a plain ``prompt: str`` cannot inject extra ids safely and
    # leave this False. The UI's prepend chip-input is gated on this.
    supports_prepend_token_ids: bool = False
    # ``True`` when ``backend.tokenize(text)`` returns a real id list
    # (HF / llamacpp-py natively; openai_compat once its per-model HF
    # tokenizer.json has been fetched). Drives the live token preview
    # under the prompt textarea in the Decode workbench.
    supports_local_tokenize: bool = False
    # ``True`` for backends that are registered but inert. Currently set
    # only on chat-only OpenAI-compat providers (NIM / OpenRouter) until
    # proper chat-mode UI lands. The web layer rejects generate-stream
    # requests against such backends with a 400, and the frontend picker
    # marks the option disabled with ``notes`` as the tooltip.
    generation_disabled: bool = False


# --------------------------------------------------------------------------- #
# /v1/info
# --------------------------------------------------------------------------- #
class InfoResponse(BaseModel):
    """Bundle the client fetches once at construction.

    ``engine_version`` is the package ``__version__``; ``backend_kind`` is
    the short name the server was started with (``"hf"`` /
    ``"llamacpp-py"``). Capabilities are echoed verbatim from the loaded
    backend so the client can adapt its UI without making more calls.

    ``capabilities`` is ``None`` when no model is currently loaded (a
    server started with ``--no-preload`` or whose model load failed): the
    server can still describe itself (kind + version + state) even with an
    empty slot. ``state`` mirrors :class:`ServerStatus.state` so a client
    that only fetches ``/v1/info`` still knows whether the slot is live.
    """

    capabilities: WireCapabilities | None = None
    engine_version: str
    backend_kind: str
    loaded_model: str | None = None
    state: str = "ready"


# --------------------------------------------------------------------------- #
# /v1/status, /v1/models, /v1/reload (swappable model slot)
# --------------------------------------------------------------------------- #
class ServerStatus(BaseModel):
    """Live state of the server's single model slot.

    ``state`` is one of ``"empty"`` (no model ever loaded / unloaded),
    ``"loading"`` (a build is in progress on a background thread),
    ``"ready"`` (a model is loaded and serving), or ``"error"`` (the most
    recent load failed -- ``error`` carries the message). ``loaded_model``
    is the id/path of the model currently in the slot (``None`` unless
    ``state == "ready"``). ``capabilities`` is populated only when ready so
    the client can refresh its capability envelope after a swap.
    """

    backend_kind: str
    state: str
    loaded_model: str | None = None
    error: str | None = None
    capabilities: WireCapabilities | None = None


class ServerModelEntry(BaseModel):
    """One selectable model on the host: an opaque ``id`` + a short label.

    For ``llamacpp-py`` the ``id`` is the absolute GGUF path and ``label``
    its filename stem; for ``hf`` both are the HuggingFace model id. The
    ``id`` is what the client POSTs back to ``/v1/reload``.
    """

    id: str
    label: str


class ServerModelList(BaseModel):
    """Catalogue of models the host can load into its current slot kind."""

    backend_kind: str
    models: list[ServerModelEntry] = Field(default_factory=list)
    note: str = ""


class ReloadRequest(BaseModel):
    """Body for ``POST /v1/reload``.

    ``model`` is the id/path to load (one of the ``/v1/models`` entries, or
    any value the backend's builder accepts). ``None`` re-loads the backend
    kind's configured default model.
    """

    model: str | None = None


# --------------------------------------------------------------------------- #
# REST endpoints (tokenize / detokenize / piece / next / score / verify)
# --------------------------------------------------------------------------- #
class TokenizeRequest(BaseModel):
    text: str


class TokenizeResponse(BaseModel):
    ids: list[int]


class DetokenizeRequest(BaseModel):
    ids: list[int]


class DetokenizeResponse(BaseModel):
    text: str


class PieceRequest(BaseModel):
    id: int


class PieceResponse(BaseModel):
    text: str


class SpecialToken(BaseModel):
    id: int
    text: str


class SpecialTokensResponse(BaseModel):
    tokens: list[SpecialToken]


class NextDistributionRequest(BaseModel):
    ids: list[int]
    top_k: int = 8


class ScorePromptRequest(BaseModel):
    prompt: str
    top_k: int = 8
    watch_ids: list[int] = Field(default_factory=list)
    # Token ids to splice in BEFORE the tokenized prompt. The backend
    # scores the combined sequence as one long context, so the first
    # ``len(prepend_token_ids)`` rows of the response correspond to
    # these injected tokens. The motivating use case is "predict
    # position 0 from BOS": without an injected token the very first
    # row of a score_prompt response would have no prior context to
    # condition the model on and the upstream can't produce a real
    # distribution there. Backends that can't safely inject extra ids
    # (cloud providers that tokenize server-side from a plain
    # prompt string) raise NotImplementedError when this is non-empty;
    # the web layer should check capabilities.supports_prepend_token_ids
    # before populating it.
    prepend_token_ids: list[int] = Field(default_factory=list)


class ScorePromptResponse(BaseModel):
    steps: list[WireStepResult]


class VerifyGreedyRequest(BaseModel):
    context_ids: list[int]
    draft_ids: list[int]


class VerifyGreedyResponse(BaseModel):
    accepted: int
    correction: WireTokenCandidate | None = None


# --------------------------------------------------------------------------- #
# /v1/generate/stream (SSE)
# --------------------------------------------------------------------------- #
class SamplerSpec(BaseModel):
    """Server-side sampler config. ``name`` selects the builtin in
    :mod:`decoding_sandbox.core.samplers`; ``params`` are forwarded as
    keyword arguments to its builder.

    Custom samplers (``--sampler custom --custom-file ...``) cannot run on
    the server (no remote code execution), so the CLI falls back to the
    per-step ``next_distribution`` loop in that case and never POSTs to
    this endpoint with ``name="custom"``.
    """

    name: str
    params: dict[str, float | int | None] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    prompt: str
    sampler: SamplerSpec
    max_tokens: int = 20
    top_k: int = 50
    stop_ids: list[int] = Field(default_factory=list)
    seed: int = 0
    respect_eos: bool = True
    # Token ids whose per-step probability the caller wants to track
    # even when they fall outside the returned top-k. The dsbx-serve
    # engine forwards them to ``Backend.next_distribution(... watch_ids=)``;
    # full-vocab backends (HF, llamacpp_py) read EXACT values from the
    # same forward-pass tensor, top-k-only backends fall back to
    # "found in top-k or rank=-1/NaN". Empty list disables the feature.
    watch_ids: list[int] = Field(default_factory=list)
    # Token ids the engine should treat as already-generated PREFIX
    # AFTER the prompt -- the model sees ``tokenize(prompt) + prefix_token_ids``
    # as one continuous sequence and starts generating from there. Powers
    # the unified workbench's "manual decoding" mode without server-side
    # session state: the browser holds the picks and resends the growing
    # id list on every pick. Empty list = no prefix (the historical
    # behaviour).
    prefix_token_ids: list[int] = Field(default_factory=list)


class WireKeptEntry(BaseModel):
    """One entry of ``SamplerDecision.kept`` -- a kept token id + its
    renormalized probability after sampler filtering."""

    token_id: int
    prob: float


class WireSamplerDecision(BaseModel):
    """Mirror of :class:`decoding_sandbox.core.samplers.SamplerDecision`."""

    token_id: int
    token_text: str
    kept: list[WireKeptEntry] = Field(default_factory=list)
    greedy_token_id: int | None = None
    note: str = ""


class WireGenStep(BaseModel):
    """Mirror of :class:`decoding_sandbox.core.engine.GenStep`."""

    step: int
    tokens_before: list[int]
    step_result: WireStepResult
    decision: WireSamplerDecision
    stop_reason: str | None = None


class StepEvent(BaseModel):
    """SSE payload for a single decoding step."""

    event: str = "step"
    step: WireGenStep


class DoneEvent(BaseModel):
    """Terminating SSE payload (always emitted, even on errors)."""

    event: str = "done"
    stop_reason: str | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# Conversions (in-memory dataclass -> wire model)
# --------------------------------------------------------------------------- #
def _safe_logprob(v) -> float | None:
    """NaN/inf -> ``None`` on the wire; finite floats pass through.

    Used by every candidate conversion so the standards-compliant
    ``logprob: float | None`` schema actually serializes cleanly through
    ``json.dumps``. The client mirror in
    ``decoding_sandbox/backends/remote.py`` does the reverse.
    """
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def candidate_to_wire(c) -> WireTokenCandidate:
    smc = getattr(c, "sampling_mask_count", None)
    return WireTokenCandidate(
        token_id=int(c.token_id),
        text=c.text,
        logprob=_safe_logprob(c.logprob),
        rank=int(c.rank),
        is_special=bool(c.is_special),
        sampling_mask_count=int(smc) if smc is not None else None,
    )


def step_to_wire(s) -> WireStepResult:
    return WireStepResult(
        position=int(s.position),
        candidates=[candidate_to_wire(c) for c in s.candidates],
        is_full_vocab=bool(s.is_full_vocab),
        chosen=candidate_to_wire(s.chosen) if s.chosen is not None else None,
        context_text=s.context_text,
        watched=[
            WireWatched(token_id=int(tid), candidate=candidate_to_wire(cand))
            for tid, cand in (s.watched or {}).items()
        ],
    )


def capabilities_to_wire(caps) -> WireCapabilities:
    return WireCapabilities(
        name=caps.name,
        full_vocab=bool(caps.full_vocab),
        prompt_logprobs=bool(caps.prompt_logprobs),
        max_top_logprobs=int(caps.max_top_logprobs),
        can_force_token=bool(caps.can_force_token),
        notes=caps.notes or "",
        eos_token_ids=list(caps.eos_token_ids or ()),
        bos_token_ids=list(getattr(caps, "bos_token_ids", ()) or ()),
        supports_ignore_eos=bool(getattr(caps, "supports_ignore_eos", False)),
        supports_perf_metrics=bool(getattr(caps, "supports_perf_metrics", False)),
        supports_service_tier=bool(getattr(caps, "supports_service_tier", False)),
        supports_sampling_mask=bool(getattr(caps, "supports_sampling_mask", False)),
        supports_raw_output=bool(getattr(caps, "supports_raw_output", False)),
        supports_logit_bias=bool(getattr(caps, "supports_logit_bias", False)),
        supports_combined_echo_stream=bool(
            getattr(caps, "supports_combined_echo_stream", False)
        ),
        supports_prepend_token_ids=bool(
            getattr(caps, "supports_prepend_token_ids", False)
        ),
        supports_local_tokenize=bool(
            getattr(caps, "supports_local_tokenize", False)
        ),
        generation_disabled=bool(getattr(caps, "generation_disabled", False)),
    )


def decision_to_wire(d) -> WireSamplerDecision:
    return WireSamplerDecision(
        token_id=int(d.token_id),
        token_text=d.token_text,
        kept=[
            WireKeptEntry(token_id=int(cand.token_id), prob=float(prob))
            for cand, prob in (d.kept or [])
        ],
        greedy_token_id=(int(d.greedy_token_id) if d.greedy_token_id is not None else None),
        note=d.note or "",
    )


def genstep_to_wire(gs) -> WireGenStep:
    return WireGenStep(
        step=int(gs.step),
        tokens_before=[int(t) for t in gs.tokens_before],
        step_result=step_to_wire(gs.step_result),
        decision=decision_to_wire(gs.decision),
        stop_reason=gs.stop_reason,
    )


__all__ = [
    "WireTokenCandidate",
    "WireWatched",
    "WireStepResult",
    "WireCapabilities",
    "InfoResponse",
    "ServerStatus",
    "ServerModelEntry",
    "ServerModelList",
    "ReloadRequest",
    "TokenizeRequest",
    "TokenizeResponse",
    "DetokenizeRequest",
    "DetokenizeResponse",
    "PieceRequest",
    "PieceResponse",
    "NextDistributionRequest",
    "ScorePromptRequest",
    "ScorePromptResponse",
    "VerifyGreedyRequest",
    "VerifyGreedyResponse",
    "SamplerSpec",
    "GenerateRequest",
    "WireKeptEntry",
    "WireSamplerDecision",
    "WireGenStep",
    "StepEvent",
    "DoneEvent",
    "candidate_to_wire",
    "step_to_wire",
    "capabilities_to_wire",
    "decision_to_wire",
    "genstep_to_wire",
]
