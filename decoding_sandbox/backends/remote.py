"""HTTP client backend that talks to a ``dsbx serve`` instance.

The client lives wherever the TUI runs (the client today, a future
browser tomorrow) and the server lives wherever the model is loaded
(``dsbx-host`` with the P40). This module is the dsbx-host <-> client bridge.

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


class RemoteBackend(Backend):
    """``Backend`` over HTTP. Talks to a ``dsbx serve`` instance."""

    # Marker attribute the CLI's cmd_generate uses to decide whether to
    # call ``stream_generate`` instead of the per-step engine loop. Kept
    # as a class attribute (rather than a method check) so the dispatch
    # remains obvious in code review.
    supports_remote_stream: bool = True

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
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
        info = self._get_info()
        self._capabilities = _capabilities_from_dict(info["capabilities"])
        self._backend_kind: str = str(info.get("backend_kind", "unknown"))
        self._loaded_model: str | None = info.get("loaded_model")
        self._engine_version: str = str(info.get("engine_version", "?"))
        # Tiny memoizer so the manual TUI doesn't fan out a piece request
        # per token per render. ``piece`` is called O(top_k) times for
        # every step; without caching every render becomes a noticeable
        # network burst.
        self._piece_cache: dict[int, str] = {}

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

    def next_distribution(self, token_ids: list[int], top_k: int) -> StepResult:
        body = {"ids": [int(i) for i in token_ids], "top_k": int(top_k)}
        data = self._post("/v1/next_distribution", body)
        return _step_from_dict(data)

    def score_prompt(
        self, prompt: str, top_k: int, watch_ids: list[int] | None = None
    ) -> list[StepResult]:
        body = {
            "prompt": prompt,
            "top_k": int(top_k),
            "watch_ids": [int(i) for i in (watch_ids or [])],
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
    ) -> Iterator[GenStep]:
        """POST to ``/v1/generate/stream`` and re-yield each step as a ``GenStep``.

        The generator runs the request inside an ``httpx.stream`` context;
        the connection stays open for the whole decode, and each SSE
        ``data:`` frame is parsed into a fresh ``GenStep``. The trailing
        ``done`` event terminates the iterator -- if it carries an
        ``error`` field the iterator raises :class:`RemoteBackendError`
        with the server's message, which surfaces cleanly in the TUI.
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
        }
        try:
            with self._client.stream("POST", "/v1/generate/stream", json=body) as r:
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
                for event in _iter_sse_events(r.iter_lines()):
                    kind = event.get("event")
                    if kind == "step":
                        yield _genstep_from_dict(event["step"])
                    elif kind == "done":
                        err = event.get("error")
                        if err:
                            raise RemoteBackendError(f"server reported error: {err}")
                        return
                    # Unknown event kinds are silently ignored; this lets
                    # the server add new event types (e.g. "progress")
                    # without breaking older clients.
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


__all__ = ["RemoteBackend", "RemoteBackendError"]
