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
    # Per-model capability envelopes for ``family == "cloud"`` providers.
    # The OpenAI-compat backend caches one instance per model in
    # ``cloud_variants``, and each variant may report different
    # ``supports_*`` flags (Fireworks's ``gpt-oss-20b`` rejects
    # ``sampling_mask``, ``gpt-oss-120b`` accepts it) and different
    # ``bos_token_ids`` (auto-discovered from the loaded HF tokenizer).
    # Without this map the listing endpoint could only surface ONE
    # envelope per backend -- the default-model variant -- and the UI
    # would happily suggest a gpt-oss BOS for glm-5p1. Frontend looks
    # up ``models_caps[currentModel] ?? capabilities`` so models whose
    # variant isn't loaded yet still get the static fallback. Always
    # empty for remote/local families (one instance == one envelope).
    models_caps: dict[str, WireCapabilities] = Field(default_factory=dict)
    available: bool = True
    note: str = ""
    loaded_model: str | None = None
    suggested_models: list[str] = Field(default_factory=list)
    model_editable: bool = False
    # ``True`` when the backend's loaded model can be swapped at runtime
    # via ``POST /api/v1/backends/{name}/reload`` -- currently only
    # ``remote`` dsbx-serve hosts (which own a swappable model slot).
    # Cloud providers use per-request ``model`` (``model_editable``)
    # instead; local in-process engines need a process restart. The
    # frontend renders a load/reload control + live state badge for
    # reloadable backends on the Status page.
    model_reloadable: bool = False


class InfoResponse(BaseModel):
    """Bundle the frontend fetches once on app load."""

    engine_version: str
    server_label: str
    default_backend: str
    backends: list[BackendInfo]


class RemoteStatusResponse(BaseModel):
    """``GET /api/v1/backends/{name}/status`` -- live remote slot state.

    Proxies the upstream dsbx-server's ``/v1/status`` (scrubbed of any
    address/URL). ``state`` is ``empty`` / ``loading`` / ``ready`` /
    ``error``; ``loaded_model`` and ``error`` mirror the upstream. The
    frontend polls this while a load is in progress.
    """

    backend: str
    state: str
    loaded_model: str | None = None
    error: str | None = None


class ReloadModelRequest(BaseModel):
    """``POST /api/v1/backends/{name}/reload`` body. ``None`` -> host default."""

    model: str | None = None


class ModelsResponse(BaseModel):
    """``GET /api/v1/models/{name}`` response.

    Lists the model ids the named backend currently advertises. For cloud
    providers this is fetched live from the upstream catalogue (NIM /
    OpenRouter / LM Studio via OpenAI-compat ``/models``, Fireworks via
    its per-account catalogue) and cached on the middleware for
    ``cache_ttl_s`` seconds. For ``remote`` / ``local`` backends this
    returns the configured-model single-element list so the browser can
    call the same endpoint uniformly.

    ``source`` is one of:

    - ``"live"``: result of an actual upstream call this turn.
    - ``"cached"``: result of an upstream call within the TTL window.
    - ``"static"``: no network needed (remote/local backends).
    - ``"fallback"``: the upstream call failed; this is the curated list.

    ``fetched_at`` is an epoch seconds timestamp (None when
    ``source=="static"`` and there's no meaningful "when"). ``note`` is a
    short human-readable comment safe to show in the UI (no URLs).
    """

    backend: str
    models: list[str]
    source: Literal["live", "cached", "static", "fallback"]
    fetched_at: float | None = None
    cache_ttl_s: float = 0.0
    note: str = ""
    # Optional ``id -> on-disk size in bytes`` map. Populated only for
    # ``remote`` dsbx-serve hosts that report per-model ``size_bytes``
    # (every GGUF); the browser uses it to draw a determinate,
    # size-proportional model-load progress bar instead of an
    # indeterminate flicker. Empty for cloud / local / HF backends.
    model_sizes: dict[str, int] = Field(default_factory=dict)


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
    # Per-token surface form (``piece(id)`` rendered as printable text).
    # Populated when the active backend has a real local tokenizer
    # (HF / llamacpp-py / openai-compat-with-mapped-tokenizer); empty
    # otherwise so the frontend can fall back to "show ids only" without
    # an extra round trip. The browser uses this to render the live
    # token preview as the user types: one ``TokenInline`` chip per
    # entry, with ids as titles for the technically-curious.
    pieces: list[str] = []


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


class SpecialTokensRequest(BaseModel):
    backend: str
    model: str | None = None


class SpecialTokenEntry(BaseModel):
    id: int
    text: str


class SpecialTokensResponse(BaseModel):
    tokens: list[SpecialTokenEntry]


# --------------------------------------------------------------------------- #
# Generate (SSE body re-uses WireGenStep verbatim)
# --------------------------------------------------------------------------- #
#
# The legacy ``InspectRequest`` / ``InspectResponse`` / ``ResolvedWatch``
# schemas used to live above ``GenerateRequest``. They are gone -- the
# inspect endpoint was deleted in plan: Unify Decode Workbench Phase 3
# because inspect is a degenerate case of generate (``max_tokens=1 +
# include_prompt=true``). ``watch_texts`` / ``watch_ids`` / ``watch_eos``
# now live on :class:`GenerateRequest`; the frontend reconstructs human-
# readable column labels from what it sent (the round-trip
# ``ResolvedWatch`` payload is no longer needed).


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
    # Provider-specific knobs forwarded to OpenAICompatBackend.stream_native
    # when the active backend's capabilities advertise support. Ignored
    # otherwise. See ProviderConfig.supports_* fields and the Fireworks
    # docs for what each does. ``session_id`` -- a stable per-session
    # identifier -- becomes the ``x-session-affinity`` /
    # ``x-multi-turn-session-id`` headers when the provider advertises
    # session affinity; primarily used by the manual-decoding mode to
    # keep KV-cache + MoE expert routing pinned to one replica.
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    session_id: str | None = None
    # Per-request OpenAI-style logit bias map: ``{token_id: bias}``,
    # where bias is in [-100, 100]. Forwarded to the provider when
    # ``Capabilities.supports_logit_bias`` is true. Keys can arrive as
    # strings or ints from JSON; the backend coerces and validates.
    # Skipped entirely on backends that don't advertise support so a
    # stale UI knob never silently lies. Use cases this turns into one-
    # liners: banning a token (``bias=-100``), boosting a rare option
    # past top_p truncation (``bias=+5..+15``), forcing a token in a
    # constrained-grammar setup (``bias=+100``).
    logit_bias: dict[str, float] | None = None
    # ``echo_last=N`` (Fireworks; gated by
    # ``ProviderConfig.supports_echo_last``) tells the provider to
    # echo logprobs for only the LAST N prompt tokens instead of every
    # one. Saves wire bytes + parsing CPU on long prompts when the
    # user really only cares about the trailing context. ``None`` ==
    # echo the whole prompt (the historical default).
    echo_last: int | None = None
    # Token ids the engine should treat as PREFIX after the prompt --
    # the model sees ``tokenize(prompt) + prefix_token_ids`` as one
    # continuous sequence and starts generating from there. Powers
    # the unified workbench's "manual decoding" mode: the browser
    # holds the user's picks and resends the growing id list on each
    # pick instead of carrying server-side session state. For
    # Fireworks the web layer turns the ids back into text via
    # ``backend.detokenize(...)`` before sending; for the per-step
    # engine path the engine appends them directly to the token
    # buffer. Empty list = no prefix (historical behaviour).
    prefix_token_ids: list[int] = Field(default_factory=list)
    # Token ids to splice in BEFORE the tokenized prompt -- the
    # opposite end of ``prefix_token_ids`` (which historically
    # acts as a SUFFIX-after-prompt for manual decoding). Sent to
    # ``backend.score_prompt(..., prepend_token_ids=...)`` so the
    # prompt-logits frame includes the BOS-conditioned distribution
    # for the user's first prompt token (otherwise unscorable: an
    # autoregressive model has nothing to predict from at position
    # 0 without prior context). Gated by
    # ``Capabilities.supports_prepend_token_ids``; the UI's
    # "fill BOS" helper drops the model's known BOS ids in here.
    # Backends that can't handle non-empty values (cloud providers
    # that tokenize server-side from a plain prompt string) raise
    # NotImplementedError; the web layer reports that as a 400.
    prepend_token_ids: list[int] = Field(default_factory=list)
    # Token ids whose per-step probability the caller wants to track
    # even when they fall outside the returned top-k. Forwarded to
    # every per-token GenStep (and to the prompt-echo StepResults
    # when ``include_prompt`` is set). Full-vocab backends return
    # exact values; top-k-only backends report ``rank=-1, logprob=NaN``
    # for ids outside the chunk's top_logprobs. Replaces the watch
    # plumbing that used to live solely on the (now-deleted) inspect
    # endpoint.
    watch_ids: list[int] = Field(default_factory=list)
    # Text strings to resolve to single tokens via the backend's
    # tokenizer (taking the first token id when a string spans
    # multiple) and merge into ``watch_ids``. Lets the user type
    # human-readable cells like " Paris" without computing ids up
    # front. Resolution happens server-side because the tokenizer is
    # backend-specific.
    watch_texts: list[str] = Field(default_factory=list)
    # When true, every id in ``Capabilities.eos_token_ids`` is added
    # as a watched cell. Saves the user from having to know the
    # model's EOS id; same UX the historical inspect endpoint had.
    watch_eos: bool = False


class StepEvent(BaseModel):
    event: Literal["step"] = "step"
    step: WireGenStep


class PromptScoreEvent(BaseModel):
    """Optional first frame of a generate stream when ``include_prompt`` is set.

    Carries one :class:`WireStepResult` per prompt token (the per-position
    distribution the model would have predicted) plus the same
    ``is_full_vocab`` / ``prompt_logprobs`` / ``note`` flags the old
    inspect endpoint returned. With the inspect endpoint deleted (plan:
    Unify Decode Workbench Phase 3), this is the canonical "show me the
    prompt logits" wire shape.
    """

    event: Literal["prompt_score"] = "prompt_score"
    steps: list[WireStepResult]
    is_full_vocab: bool
    prompt_logprobs: bool
    note: str = ""


class PerfMetricsEvent(BaseModel):
    """Server-side performance metrics, emitted before ``usage`` / ``done``.

    Populated only by providers that advertise
    ``Capabilities.supports_perf_metrics`` and that returned a
    ``perf_metrics`` block in the response body (or final streaming
    chunk). See Fireworks' /v1/completions docs for the full schema --
    we wrap it in an opaque ``metrics`` dict so adding a new key on the
    upstream doesn't force a wire-schema bump.

    Typical fields rendered by the UI:

    - ``server-time-to-first-token``: TTFT in seconds
    - ``prompt-tokens`` / ``cached-prompt-tokens``: prompt size + cache
      hit count
    - ``prefill-duration`` / ``generation-duration``: time split between
      forward-pass phases
    - ``speculation-acceptance``: per-position acceptance rates when
      Fireworks ran speculative decoding under the hood
    - ``backend-host``: which replica served us (dedicated deployments
      only); useful when investigating "this run was slow but the next
      was fast" on a multi-replica deployment.
    """

    event: Literal["perf"] = "perf"
    metrics: dict = Field(default_factory=dict)


class RawOutputEvent(BaseModel):
    """Server-side "what the model actually saw" diagnostics.

    Emitted between ``perf`` and ``usage`` (so order is ``step* ->
    perf? -> raw_output? -> usage -> done``) when the provider
    advertises ``supports_raw_output`` and returned a non-empty
    ``raw_output`` block. The payload is a verbatim copy of the
    provider's dict so the browser's "what the model saw" panel can
    render every key the provider chose to include without us having
    to type each one. Typical keys (Fireworks):

    - ``prompt_fragments`` -- the prompt broken into the chunks the
      templating engine fed to the model. Sanity-check for chat
      templates that silently drop role tags.
    - ``prompt_token_ids`` -- the actual tokenized prompt including
      injected BOS / system tokens. Surfaces silent tokenizer
      mismatches between the UI text and what the model saw.
    - ``grammar`` -- compiled grammar object when response_format /
      json mode is active; tells you which constrained-decoding FSM
      ran.
    """

    event: Literal["raw_output"] = "raw_output"
    payload: dict = Field(default_factory=dict)


class UsageEvent(BaseModel):
    """Per-run resource accounting, emitted immediately BEFORE ``done``.

    Lets the UI surface how heavy a generate call actually was: number
    of HTTP requests against an upstream provider (so the spamming-the-
    cloud pattern that historically tripped Fireworks's per-account RPS
    limit on glm-5p2 is visible at a glance), and the prompt/completion
    token totals reported by the provider's billing layer when available.

    Token fields are nullable so backends that can't measure a quantity
    leave them ``None`` and the UI renders ``—``. ``notes`` carries
    advisory lines a backend may want the user to see -- for example,
    "this cloud provider always halts on EOS" when the caller asked
    for ``respect_eos=False`` against a Fireworks model.

    The frame is intentionally separate from ``done`` so existing
    consumers that key off the terminal event don't have to learn a
    new shape; the wire order is unchanged otherwise.
    """

    event: Literal["usage"] = "usage"
    requests: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    notes: list[str] = Field(default_factory=list)


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
    "RemoteStatusResponse",
    "ReloadModelRequest",
    "ModelsResponse",
    "TokenizeRequest",
    "TokenizeResponse",
    "DetokenizeRequest",
    "DetokenizeResponse",
    "PieceRequest",
    "PieceResponse",
    "GenerateRequest",
    "StepEvent",
    "PromptScoreEvent",
    "PerfMetricsEvent",
    "RawOutputEvent",
    "UsageEvent",
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
