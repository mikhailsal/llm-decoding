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
    """

    name: str
    label: str
    family: Literal["remote", "cloud", "local"]
    capabilities: WireCapabilities | None = None
    available: bool = True
    note: str = ""


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


class TokenizeResponse(BaseModel):
    ids: list[int]


class DetokenizeRequest(BaseModel):
    backend: str
    ids: list[int]


class DetokenizeResponse(BaseModel):
    text: str


class PieceRequest(BaseModel):
    backend: str
    id: int


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


class StepEvent(BaseModel):
    event: Literal["step"] = "step"
    step: WireGenStep


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


class ManualSnapshot(BaseModel):
    """Single response shape every manual endpoint returns.

    Bundles everything the UI needs to re-render after any mutation: the
    server-side ``session_id``, current prompt + generated text, the next-token
    distribution at the current cursor, and the active ``top_k``.
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


# --------------------------------------------------------------------------- #
# Spec (SSE)
# --------------------------------------------------------------------------- #


class SpecRequest(BaseModel):
    target_backend: str
    draft_backend: str
    prompt: str
    gamma: int = 4
    max_tokens: int = 24


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
