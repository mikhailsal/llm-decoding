"""HTTP client backend that talks to a ``dsbx serve`` instance.

The client lives wherever the TUI runs (the client today, a future
browser tomorrow) and the server lives wherever the model is loaded
(``dsbx-host`` with the GPU). This module is the dsbx-host <-> client bridge.

Design notes:

- ``Capabilities`` is cached from ``/v1/info`` at construction. Heavy
  callers (``inspect``/``generate``) read it many times and we don't want
  a round trip per access.
- Every other ``Backend`` method maps to one HTTP call. JSON is parsed
  into the existing dataclasses from :mod:`decoding_sandbox.core.types`,
  not pydantic models -- the client deliberately has no pydantic
  dependency so a client install can stay light.
- ``stream_generate`` consumes the SSE endpoint and re-yields ``GenStep``
  objects with the same shape the in-process ``core.engine.generate``
  produces, so ``cmd_generate`` can call either with one branch.
- An ``httpx.Client`` instance is injectable for tests (``TestClient`` or
  ``MockTransport``); construction defaults to a real client against the
  configured ``base_url``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.engine import GenStep
from decoding_sandbox.core.samplers import SamplerDecision
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate


class RemoteBackendError(RuntimeError):
    """Raised when the server returns a non-2xx response or invalid JSON."""


class RemoteStreamTimeout(RemoteBackendError):
    """Raised when the upstream remote server stops sending bytes mid-stream.

    Kept as a dedicated subclass so callers (notably ``stream_generate``
    in the web layer) can present a friendly "server stopped responding"
    message instead of the raw httpx exception. Inherits from
    :class:`RemoteBackendError` so any existing
    ``except RemoteBackendError`` blocks continue to swallow it.
    """


class RemoteBackend(Backend):
    """``Backend`` over HTTP. Talks to a ``dsbx serve`` instance."""

    # Marker attribute the CLI's cmd_generate uses to decide whether to
    # call ``stream_generate`` instead of the per-step engine loop. Kept
    # as a class attribute (rather than a method check) so the dispatch
    # remains obvious in code review.
    supports_remote_stream: bool = True

    # Per-frame timeout applied to ``stream_generate``. Separate from the
    # outer ``timeout`` (which governs connect / non-stream requests like
    # /v1/info or /v1/score_prompt) because for a stream the meaningful
    # liveness signal is "the next SSE frame arrives within N seconds",
    # not "the whole response finishes within N seconds". Defaults to
    # 45 s, which is generous enough for a single GPU generate step on
    # a long prompt but tight enough that a fully hung server surfaces
    # as ``RemoteStreamTimeout`` before the user gives up and reloads.
    # Bumped via ``stream_read_timeout=`` if a slow deployment trips it.
    DEFAULT_STREAM_READ_TIMEOUT: float = 45.0

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        stream_read_timeout: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        if client is not None:
            # An explicit client overrides the transport kwarg -- the
            # test path uses this to inject a ``TestClient`` or
            # ``MockTransport`` directly. The web layer always uses the
            # ``transport=`` form so logging is uniform across backends.
            self._client = client
        else:
            client_kwargs: dict[str, object] = {"base_url": self.base_url, "timeout": timeout}
            if transport is not None:
                client_kwargs["transport"] = transport
            self._client = httpx.Client(**client_kwargs)  # type: ignore[arg-type]
        self._stream_read_timeout: float = (
            float(stream_read_timeout)
            if stream_read_timeout is not None
            else self.DEFAULT_STREAM_READ_TIMEOUT
        )
        info = self._get_info()
        self._apply_info(info)
        # Tiny memoizer so the manual TUI doesn't fan out a piece request
        # per token per render. ``piece`` is called O(top_k) times for
        # every step; without caching every render becomes a noticeable
        # network burst.
        self._piece_cache: dict[int, str] = {}
        self._special_tokens_cache: list[tuple[int, str]] | None = None

    # ------------------------------------------------------------------ #
    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def backend_kind(self) -> str:
        return self._backend_kind

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_model

    @property
    def engine_version(self) -> str:
        return self._engine_version

    @property
    def state(self) -> str:
        """Upstream slot state: ``empty`` / ``loading`` / ``ready`` / ``error``.

        Snapshotted from the last ``/v1/info`` (or ``refresh_info()``).
        Older dsbx-servers that predate the swappable slot omit the field;
        they're always serving, so we default to ``ready``.
        """
        return self._state

    def _apply_info(self, info: dict) -> None:
        """Adopt a ``/v1/info`` payload, tolerating an empty/loading slot.

        A server started with ``--no-preload`` (or mid-load / post-error)
        returns ``capabilities = null``. Rather than raise -- which would
        make the whole RemoteBackend unconstructable until a model is
        loaded -- we install a neutral placeholder ``Capabilities`` so the
        handle stays usable for status/reload calls. The real envelope is
        adopted on the next ``refresh_info()`` once the slot goes ``ready``.
        """
        caps_raw = info.get("capabilities")
        if caps_raw:
            self._capabilities = _capabilities_from_dict(caps_raw)
        else:
            self._capabilities = _placeholder_capabilities()
        self._backend_kind = str(info.get("backend_kind", "unknown"))
        self._loaded_model = info.get("loaded_model")
        self._engine_version = str(info.get("engine_version", "?"))
        self._state = str(info.get("state", "ready"))

    def refresh_info(self) -> None:
        """Re-fetch ``/v1/info`` and update cached caps / loaded model / state.

        Called by the web layer after a model swap so the next
        ``/api/v1/info`` reflects the newly loaded model's capabilities and
        name instead of the stale envelope captured at construction. Also
        clears the per-token piece cache (a different model means different
        ids -> different surface forms).
        """
        self._apply_info(self._get_info())
        self._piece_cache = {}
        self._special_tokens_cache = None

    # ----------------------------------------------- model slot control
    def server_status(self) -> dict:
        """Return the upstream ``/v1/status`` payload as a plain dict."""
        return self._get("/v1/status")

    def list_server_models(self) -> list[dict]:
        """Return the host's model catalogue from ``/v1/models``.

        Each entry is ``{"id": ..., "label": ...}``. Degrades to an empty
        list against an older dsbx-server that predates the endpoint (404).
        """
        try:
            data = self._get("/v1/models")
        except RemoteBackendError:
            return []
        return list(data.get("models", []) or [])

    def reload_model(self, model: str | None) -> dict:
        """Ask the host to (re)load ``model``; returns the new status dict.

        The upstream kicks a background load and returns immediately with
        ``state == "loading"``; the caller polls ``server_status`` until it
        reaches ``ready`` / ``error``.
        """
        return self._post("/v1/reload", {"model": model})

    def unload_model(self) -> dict:
        """Ask the host to unload the current model; returns the new status dict."""
        return self._post("/v1/unload", {})

    # ------------------------------------------------------------- HTTP
    def _get(self, path: str) -> dict:
        try:
            r = self._client.get(path)
        except httpx.HTTPError as exc:
            raise RemoteBackendError(f"GET {path}: {exc}") from exc
        return _check_json(r, path)

    def _post(self, path: str, body: dict) -> dict:
        try:
            r = self._client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise RemoteBackendError(f"POST {path}: {exc}") from exc
        return _check_json(r, path)

    def _get_info(self) -> dict:
        return self._get("/v1/info")

    # ------------------------------------------------------- protocol
    def tokenize(self, text: str) -> list[int]:
        data = self._post("/v1/tokenize", {"text": text})
        return [int(i) for i in data["ids"]]

    def detokenize(self, token_ids: list[int]) -> str:
        data = self._post("/v1/detokenize", {"ids": [int(i) for i in token_ids]})
        return str(data.get("text", ""))

    def piece(self, token_id: int) -> str:
        if token_id in self._piece_cache:
            return self._piece_cache[token_id]
        data = self._post("/v1/piece", {"id": int(token_id)})
        text = str(data.get("text", ""))
        self._piece_cache[token_id] = text
        return text

    def special_tokens(self) -> list[tuple[int, str]]:
        """Proxy to the remote server's ``/v1/special_tokens``.

        Degrades to an empty list when the remote is an OLDER dsbx-server
        that predates the endpoint (404) -- the Decode workbench just won't
        render a palette for that backend instead of erroring. Result is
        cached for the lifetime of this backend handle.
        """
        if self._special_tokens_cache is not None:
            return self._special_tokens_cache
        out: list[tuple[int, str]] = []
        try:
            data = self._post("/v1/special_tokens", {})
            for entry in data.get("tokens", []) or []:
                out.append((int(entry["id"]), str(entry["text"])))
        except RemoteBackendError:
            out = []
        self._special_tokens_cache = out
        return out

    def next_distribution(self, token_ids: list[int], top_k: int) -> StepResult:
        body = {"ids": [int(i) for i in token_ids], "top_k": int(top_k)}
        data = self._post("/v1/next_distribution", body)
        return _step_from_dict(data)

    def score_prompt(
        self,
        prompt: str,
        top_k: int,
        watch_ids: list[int] | None = None,
        *,
        prepend_token_ids: Sequence[int] = (),
    ) -> list[StepResult]:
        body = {
            "prompt": prompt,
            "top_k": int(top_k),
            "watch_ids": [int(i) for i in (watch_ids or [])],
            "prepend_token_ids": [int(i) for i in (prepend_token_ids or [])],
        }
        data = self._post("/v1/score_prompt", body)
        return [_step_from_dict(s) for s in data.get("steps", [])]

    def verify_greedy(
        self, context_ids: list[int], draft_ids: list[int]
    ) -> tuple[int, TokenCandidate]:
        body = {
            "context_ids": [int(i) for i in context_ids],
            "draft_ids": [int(i) for i in draft_ids],
        }
        data = self._post("/v1/verify_greedy", body)
        correction = data.get("correction")
        if correction is None:
            # The server only returns None when its backend itself ran out
            # of continuations -- exceedingly rare, but mirror the
            # in-process verify_greedy contract by raising rather than
            # returning a fake "" candidate the caller has to special-case.
            raise RemoteBackendError(
                "verify_greedy: server returned no correction/bonus token"
            )
        return int(data["accepted"]), _candidate_from_dict(correction)

    # ------------------------------------------------- streaming generate
    def stream_generate(
        self,
        prompt: str,
        sampler_name: str,
        sampler_params: dict[str, Any] | None = None,
        *,
        max_tokens: int = 20,
        top_k: int = 50,
        stop_ids: Sequence[int] = (),
        seed: int = 0,
        respect_eos: bool = True,
        watch_ids: Sequence[int] = (),
        prefix_token_ids: Sequence[int] = (),
    ) -> Iterator[GenStep]:
        """POST to ``/v1/generate/stream`` and re-yield each step as a ``GenStep``.

        The generator runs the request inside an ``httpx.stream`` context;
        the connection stays open for the whole decode, and each SSE
        ``data:`` frame is parsed into a fresh ``GenStep``. The trailing
        ``done`` event terminates the iterator -- if it carries an
        ``error`` field the iterator raises :class:`RemoteBackendError`
        with the server's message, which surfaces cleanly in the TUI.

        ``watch_ids`` -- token ids whose per-step probability the caller
        wants to track even when they fall outside top_k. Forwarded to
        dsbx-serve via the ``watch_ids`` body field; the remote engine
        plumbs them through to :meth:`Backend.next_distribution`. A
        full-vocab remote backend (dsbx-host-py / HF / llamacpp_py) returns
        exact logprobs; a top-k-only remote would fall back to
        ``rank=-1, logprob=NaN``, but the only deployed remote shape
        today is full-vocab so the field round-trips cleanly.
        """
        body = {
            "prompt": prompt,
            "sampler": {
                "name": sampler_name,
                "params": dict(sampler_params or {}),
            },
            "max_tokens": int(max_tokens),
            "top_k": int(top_k),
            "stop_ids": [int(i) for i in stop_ids],
            "seed": int(seed),
            "respect_eos": bool(respect_eos),
            "watch_ids": [int(i) for i in watch_ids],
            # Manual-decoding picks: the dsbx-serve engine appends these
            # to the tokenized prompt before the first decode step.
            "prefix_token_ids": [int(i) for i in prefix_token_ids],
        }
        # Per-stream timeout: tight ``read`` so a hung server surfaces as
        # ``RemoteStreamTimeout`` within seconds (and lets the web layer
        # release its sync generator so Starlette's ``GeneratorExit``
        # from a disconnected browser tab can actually take effect).
        # ``write`` / ``pool`` stay short because the upload is a small
        # JSON blob; only ``connect`` is bumped to 10 s for slow LANs.
        stream_timeout = httpx.Timeout(
            connect=10.0,
            read=self._stream_read_timeout,
            write=10.0,
            pool=10.0,
        )
        try:
            with self._client.stream(
                "POST", "/v1/generate/stream", json=body, timeout=stream_timeout
            ) as r:
                if r.status_code >= 400:
                    # Read the (small) body so the server's HTTPException
                    # detail makes it back to the user.
                    try:
                        detail = r.read().decode("utf-8", errors="replace")
                    except Exception:  # noqa: BLE001
                        detail = ""
                    raise RemoteBackendError(
                        f"POST /v1/generate/stream -> HTTP {r.status_code}: {detail}"
                    )
                try:
                    for event in _iter_sse_events(r.iter_lines()):
                        kind = event.get("event")
                        if kind == "step":
                            yield _genstep_from_dict(event["step"])
                        elif kind == "done":
                            err = event.get("error")
                            if err:
                                raise RemoteBackendError(
                                    f"server reported error: {err}"
                                )
                            return
                        # Unknown event kinds are silently ignored; this lets
                        # the server add new event types (e.g. "progress")
                        # without breaking older clients.
                except GeneratorExit:
                    # Re-raise so the outer ``with`` actually closes the
                    # connection (httpx ``Response`` ``__exit__`` runs
                    # ``stream.close()`` which RSTs the socket -- that's
                    # how the upstream dsbx serve learns the client gave
                    # up and stops generating).
                    raise
        except httpx.ReadTimeout as exc:
            # Distinct, actionable error: the server is silent. Common
            # causes: the model loader is stuck, the dsbx serve event
            # loop is blocked by a prior request, or the network path
            # is one-way. The web layer surfaces this as a clean
            # ``done.error`` SSE so the browser shows it in the toast.
            raise RemoteStreamTimeout(
                f"upstream stopped sending data for >{self._stream_read_timeout:.0f}s; "
                "the remote ``dsbx serve`` may be hung or overloaded"
            ) from exc
        except httpx.HTTPError as exc:
            raise RemoteBackendError(f"stream connection error: {exc}") from exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


# --------------------------------------------------------------------------- #
# Wire -> dataclass conversions (no pydantic on the client)
# --------------------------------------------------------------------------- #
def _check_json(r: httpx.Response, path: str) -> dict:
    if r.status_code >= 400:
        # FastAPI returns ``{"detail": "..."}`` for HTTPException; surface it.
        try:
            payload = r.json()
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        except Exception:  # noqa: BLE001
            detail = r.text
        raise RemoteBackendError(f"{path} -> HTTP {r.status_code}: {detail}")
    try:
        return r.json()
    except ValueError as exc:
        raise RemoteBackendError(f"{path}: invalid JSON ({exc})") from exc


def _candidate_from_dict(d: dict) -> TokenCandidate:
    """Build a TokenCandidate from a wire dict.

    ``logprob`` is ``None`` on the wire when the server doesn't know it
    (token outside the returned top-k); we restore ``math.nan`` here so
    the rest of the codebase -- particularly the ``watch_cell`` /
    ``fmt_prob`` renderer -- keeps treating "unknown logprob" the same
    way it does for in-process backends.
    """
    raw_lp = d.get("logprob")
    logprob = float(raw_lp) if raw_lp is not None else math.nan
    smc = d.get("sampling_mask_count")
    return TokenCandidate(
        token_id=int(d["token_id"]),
        text=str(d.get("text", "")),
        logprob=logprob,
        rank=int(d.get("rank", -1)),
        is_special=bool(d.get("is_special", False)),
        sampling_mask_count=int(smc) if smc is not None else None,
    )


def _step_from_dict(d: dict) -> StepResult:
    candidates = [_candidate_from_dict(c) for c in d.get("candidates", [])]
    chosen_raw = d.get("chosen")
    chosen = _candidate_from_dict(chosen_raw) if chosen_raw is not None else None
    watched: dict[int, TokenCandidate] = {}
    for entry in d.get("watched", []) or []:
        tid = int(entry["token_id"])
        watched[tid] = _candidate_from_dict(entry["candidate"])
    return StepResult(
        position=int(d["position"]),
        candidates=candidates,
        is_full_vocab=bool(d.get("is_full_vocab", False)),
        chosen=chosen,
        context_text=d.get("context_text"),
        watched=watched,
    )


def _placeholder_capabilities() -> Capabilities:
    """A neutral envelope for a remote whose slot has no model loaded yet.

    Everything is conservatively off / empty so the UI doesn't advertise a
    feature against a backend that can't currently serve. Replaced by the
    real envelope on the first ``refresh_info()`` after the slot goes
    ``ready``.
    """
    return Capabilities(
        name="remote:(no model loaded)",
        full_vocab=False,
        prompt_logprobs=False,
        max_top_logprobs=0,
        can_force_token=False,
        notes="no model loaded on the remote host",
    )


def _capabilities_from_dict(d: dict) -> Capabilities:
    return Capabilities(
        name=str(d["name"]),
        full_vocab=bool(d["full_vocab"]),
        prompt_logprobs=bool(d["prompt_logprobs"]),
        max_top_logprobs=int(d["max_top_logprobs"]),
        can_force_token=bool(d.get("can_force_token", False)),
        notes=str(d.get("notes", "")),
        eos_token_ids=tuple(int(i) for i in d.get("eos_token_ids", [])),
        supports_ignore_eos=bool(d.get("supports_ignore_eos", False)),
        supports_perf_metrics=bool(d.get("supports_perf_metrics", False)),
        supports_service_tier=bool(d.get("supports_service_tier", False)),
        supports_sampling_mask=bool(d.get("supports_sampling_mask", False)),
        supports_raw_output=bool(d.get("supports_raw_output", False)),
        supports_logit_bias=bool(d.get("supports_logit_bias", False)),
        supports_combined_echo_stream=bool(
            d.get("supports_combined_echo_stream", False)
        ),
        bos_token_ids=tuple(int(i) for i in d.get("bos_token_ids", [])),
        supports_prepend_token_ids=bool(d.get("supports_prepend_token_ids", False)),
        supports_local_tokenize=bool(d.get("supports_local_tokenize", False)),
    )


def _decision_from_dict(d: dict, candidates: list[TokenCandidate]) -> SamplerDecision:
    """Rebuild a ``SamplerDecision``; ``kept`` is joined to ``candidates``."""
    by_id = {c.token_id: c for c in candidates}
    kept_pairs: list[tuple[TokenCandidate, float]] = []
    for entry in d.get("kept", []) or []:
        tid = int(entry["token_id"])
        cand = by_id.get(tid)
        if cand is None:
            # The server kept a token outside the returned candidate list
            # (can happen if a future server sends a slimmer candidate
            # set than the kept list). Fall back to a stub so client
            # rendering still works without a KeyError.
            cand = TokenCandidate(tid, "", float("nan"), -1)
        kept_pairs.append((cand, float(entry["prob"])))
    greedy = d.get("greedy_token_id")
    return SamplerDecision(
        token_id=int(d["token_id"]),
        token_text=str(d.get("token_text", "")),
        kept=kept_pairs,
        greedy_token_id=int(greedy) if greedy is not None else None,
        note=str(d.get("note", "")),
    )


def _genstep_from_dict(d: dict) -> GenStep:
    step_result = _step_from_dict(d["step_result"])
    decision = _decision_from_dict(d["decision"], step_result.candidates)
    return GenStep(
        step=int(d["step"]),
        tokens_before=[int(t) for t in d.get("tokens_before", [])],
        step_result=step_result,
        decision=decision,
        stop_reason=d.get("stop_reason"),
    )


# --------------------------------------------------------------------------- #
# SSE parser (line-based; one event per ``data:`` frame separated by a blank
# line, exactly what dsbx server emits).
# --------------------------------------------------------------------------- #
def _iter_sse_events(lines: Iterator[str]) -> Iterator[dict]:
    """Parse SSE ``data:`` frames out of an iterator of raw lines.

    Each event is one or more consecutive ``data: ...`` lines terminated
    by a blank line. ``httpx`` strips the trailing ``\\n`` for us, so the
    blank-line sentinel arrives as an empty string. Other SSE fields
    (``event:``, ``id:``, ``retry:``) are ignored -- the server only uses
    ``data:`` and the payload itself carries the event kind in the JSON.
    """
    buffer: list[str] = []
    for raw in lines:
        line = raw
        if line == "":
            if buffer:
                payload = "\n".join(buffer)
                buffer.clear()
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    # Skip malformed frames rather than aborting -- one
                    # bad payload shouldn't abandon the rest of the
                    # stream.
                    continue
            continue
        if line.startswith(":"):
            # Comment / heartbeat -- ignore.
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].lstrip())
    # End of stream with unterminated buffer (server crashed mid-frame).
    if buffer:
        payload = "\n".join(buffer)
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            return


__all__ = ["RemoteBackend", "RemoteBackendError", "RemoteStreamTimeout"]
