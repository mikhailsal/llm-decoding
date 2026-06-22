"""Pydantic wire schemas for the dsbx *web* middleware.

These models are the public, browser-facing wire format. They deliberately
do NOT mirror :mod:`decoding_sandbox.core.config` -- a public ``BackendInfo``
exposes a friendly label and capability flags, never the ``base_url`` of a
remote dsbx server or the ``api_key_env`` of a cloud provider.

Where the wire shape happens to coincide with the existing in-process server's
schemas (``WireStepResult``, ``WireGenStep``, ``WireCapabilities``), we re-use
the converters from :mod:`decoding_sandbox.server.schemas` rather than
duplicate them. That way every change to the in-memory dataclasses propagates
to both servers from a single source of truth.

Three design notes worth keeping in mind:

- ``BackendInfo`` is intentionally minimal: name + label + family + capabilities.
  No URLs, no credential references, no model paths.
- Manual sessions live entirely on the server (the browser holds only a
  ``session_id``); ``ManualSnapshot`` is what every manual-mode endpoint
  returns so the UI only has to know one shape.
- ``GenerateRequest`` re-uses :class:`decoding_sandbox.server.schemas.SamplerSpec`
  unchanged -- the server-side sampler builder is the same.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from decoding_sandbox.server.schemas import (
    SamplerSpec,
    WireCapabilities,
    WireGenStep,
    WireStepResult,
    WireTokenCandidate,
)

# --------------------------------------------------------------------------- #
# Backend listing (the "what can I do?" endpoint)
# --------------------------------------------------------------------------- #


class BackendInfo(BaseModel):
    """One row of ``GET /api/v1/info``'s backend list.

    The ``family`` field is the broad class the browser uses for grouping
    (remote dsbx servers vs. cloud providers vs. local in-process engines).
    It is NOT a URL or anything identifying the server's location.

    ``loaded_model`` is the model currently in use (or the default for
    deferred-load backends). For ``remote`` it's pulled from the upstream
    ``/v1/info``; for ``local`` engines it's the configured model; for
    ``cloud`` it's the provider's ``default_model``. The browser displays
    this so the user always sees *what* they're talking to.

    ``suggested_models`` is the list of picks the UI offers in its model
    dropdown. For ``cloud`` providers we read it from
    ``[providers.NAME].models`` (defaulting to ``[default_model]``); for
    ``remote`` / ``local`` it's a single-element list with the
    ``loaded_model`` -- the browser disables the dropdown in those cases
    because changing it would require restarting a heavy process.

    ``model_editable`` is the UX hint: ``true`` means the browser lets the
    user pick a different model per request and the middleware actually
    honors it. Only cloud providers set this true today.
    """

    name: str
    label: str
    family: Literal["remote", "cloud", "local"]
    capabilities: WireCapabilities | None = None
    available: bool = True
    note: str = ""
    loaded_model: str | None = None
    suggested_models: list[str] = Field(default_factory=list)
    model_editable: bool = False


class InfoResponse(BaseModel):
    """Bundle the frontend fetches once on app load."""

    engine_version: str
    server_label: str
    default_backend: str
    backends: list[BackendInfo]


# --------------------------------------------------------------------------- #
# Tokenization & misc per-backend RPCs
# --------------------------------------------------------------------------- #


class TokenizeRequest(BaseModel):
    backend: str
    text: str
    # Optional model override (honored for cloud providers; ignored elsewhere).
    model: str | None = None


class TokenizeResponse(BaseModel):
    ids: list[int]


class DetokenizeRequest(BaseModel):
    backend: str
    ids: list[int]
    model: str | None = None


class DetokenizeResponse(BaseModel):
    text: str


class PieceRequest(BaseModel):
    backend: str
    id: int
    model: str | None = None


class PieceResponse(BaseModel):
    text: str


# --------------------------------------------------------------------------- #
# Inspect
# --------------------------------------------------------------------------- #


class InspectRequest(BaseModel):
    backend: str
    prompt: str
    top_k: int = 8
    watch_texts: list[str] = Field(default_factory=list)
    watch_ids: list[int] = Field(default_factory=list)
    watch_eos: bool = False
    model: str | None = None


class ResolvedWatch(BaseModel):
    """One resolved watch column -- enough for the UI to render the header.

    ``source`` records which user input produced this column (text/id/eos)
    so the UI can show the right header style. ``label`` is the literal
    string the CLI would have used, so the renderer logic stays parallel.
    """

    label: str
    token_id: int
    source: Literal["text", "id", "eos"]
    piece: str = ""


class InspectResponse(BaseModel):
    """``score_prompt`` output plus the resolved watch column metadata."""

    steps: list[WireStepResult]
    watches: list[ResolvedWatch]
    is_full_vocab: bool
    prompt_logprobs: bool
    note: str = ""


# --------------------------------------------------------------------------- #
# Generate (SSE body re-uses WireGenStep verbatim)
# --------------------------------------------------------------------------- #


class GenerateRequest(BaseModel):
    backend: str
    prompt: str
    sampler: SamplerSpec
    max_tokens: int = 20
    top_k: int = 50
    stop_texts: list[str] = Field(default_factory=list)
    stop_ids: list[int] = Field(default_factory=list)
    seed: int = 0
    respect_eos: bool = True
    model: str | None = None
    # When true, the stream emits a ``prompt_score`` event with the
    # per-prompt-token distribution BEFORE the first ``step`` event. The
    # browser appends those rows to the table as if they were extra steps,
    # giving the user a one-stop view of "everything the model knew about
    # the prompt, plus everything it would emit next". Chat-only backends
    # that don't support prompt logprobs fall back to a single next-token
    # row, matching the inspect-page convention.
    include_prompt: bool = False


class StepEvent(BaseModel):
    event: Literal["step"] = "step"
    step: WireGenStep


class PromptScoreEvent(BaseModel):
    """Optional first frame of a generate stream when ``include_prompt`` is set.

    The shape is intentionally identical to the body of
    :class:`InspectResponse` so a browser that already knows how to render
    one row of inspect can render these rows too -- no extra component.
    """

    event: Literal["prompt_score"] = "prompt_score"
    steps: list[WireStepResult]
    is_full_vocab: bool
    prompt_logprobs: bool
    note: str = ""


class DoneEvent(BaseModel):
    event: Literal["done"] = "done"
    stop_reason: str | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# Manual sessions
# --------------------------------------------------------------------------- #


class ManualCreateRequest(BaseModel):
    backend: str
    prompt: str
    top_k: int = 12
    model: str | None = None


class ManualSnapshot(BaseModel):
    """Single response shape every manual endpoint returns.

    Bundles everything the UI needs to re-render after any mutation: the
    server-side ``session_id``, current prompt + generated text, the next-token
    distribution at the current cursor, and the active ``top_k``. We also
    echo back a per-emitted-token ``probs`` list (one float per
    ``generated_ids``) so the browser can color the running completion
    text by token confidence without a second round trip.
    """

    session_id: str
    backend: str
    prompt: str
    prompt_ids: list[int]
    generated_ids: list[int]
    generated_text: str
    top_k: int
    distribution: WireStepResult
    can_force_token: bool
    # Linear-prob (NOT logprob) for each ``generated_ids[i]`` at the time
    # the user picked/forced it. ``None`` slots mean "we don't know" --
    # e.g. forced tokens that weren't in the top-k of the original
    # distribution. The browser uses this to color the running text the
    # same way ``/generate`` does.
    generated_probs: list[float | None] = Field(default_factory=list)
    # Per-token printable piece for each ``generated_ids[i]``. Provided so
    # the browser can render the running completion as colored chunks
    # (one ``<span>`` per token) without an extra ``/piece`` round trip.
    # Always in lockstep with ``generated_ids``.
    generated_pieces: list[str] = Field(default_factory=list)
    model: str | None = None


class ManualPickRequest(BaseModel):
    rank: int


class ManualForceRequest(BaseModel):
    """Force a token by text OR by id. Exactly one must be provided."""

    text: str | None = None
    id: int | None = None


class ManualSetTopKRequest(BaseModel):
    top_k: int


class ManualTranscript(BaseModel):
    """JSON-serializable transcript the UI can save to disk and re-load.

    Mirrors :meth:`decoding_sandbox.core.manual.ManualSession.to_dict` but
    is the explicit wire format here.
    """

    prompt: str
    backend: str
    prompt_ids: list[int]
    generated_ids: list[int]
    generated_text: str
    top_k: int
    model: str | None = None


# --------------------------------------------------------------------------- #
# Spec (SSE)
# --------------------------------------------------------------------------- #


class SpecRequest(BaseModel):
    target_backend: str
    draft_backend: str
    prompt: str
    gamma: int = 4
    max_tokens: int = 24
    target_model: str | None = None
    draft_model: str | None = None


class WireSpecRound(BaseModel):
    """Mirror of :class:`decoding_sandbox.core.speculative.SpecRound`."""

    step: int
    proposed: list[WireTokenCandidate]
    accepted: int
    correction: WireTokenCandidate | None = None
    emitted_ids: list[int] = Field(default_factory=list)


class SpecRoundEvent(BaseModel):
    event: Literal["round"] = "round"
    round: WireSpecRound


class SpecDoneEvent(BaseModel):
    event: Literal["done"] = "done"
    total_proposed: int = 0
    total_accepted: int = 0
    total_emitted: int = 0
    rounds: int = 0
    completion: str = ""
    error: str | None = None


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #


class ProbeRow(BaseModel):
    """One row of ``GET /api/v1/probe``'s response.

    ``model`` is the model name probed; ``chat_logprobs`` / ``prompt_logprobs``
    are short status strings matching the TUI's table output exactly.
    """

    provider: str
    model: str
    chat_logprobs: str
    prompt_logprobs: str


class ProbeResponse(BaseModel):
    rows: list[ProbeRow]
    fresh: bool
    cached_at: float | None = None  # epoch seconds


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    ok: bool = True
    version: str
    server_label: str


__all__ = [
    "BackendInfo",
    "InfoResponse",
    "TokenizeRequest",
    "TokenizeResponse",
    "DetokenizeRequest",
    "DetokenizeResponse",
    "PieceRequest",
    "PieceResponse",
    "InspectRequest",
    "InspectResponse",
    "ResolvedWatch",
    "GenerateRequest",
    "StepEvent",
    "PromptScoreEvent",
    "DoneEvent",
    "ManualCreateRequest",
    "ManualSnapshot",
    "ManualPickRequest",
    "ManualForceRequest",
    "ManualSetTopKRequest",
    "ManualTranscript",
    "SpecRequest",
    "WireSpecRound",
    "SpecRoundEvent",
    "SpecDoneEvent",
    "ProbeRow",
    "ProbeResponse",
    "HealthResponse",
]
