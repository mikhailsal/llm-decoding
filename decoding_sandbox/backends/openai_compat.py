"""OpenAI-compatible cloud/local backend (Fireworks / NIM / OpenRouter / LM Studio).

Cloud providers tokenize server-side and only return logprobs as *strings*, so
this backend works in text space: it assigns a stable synthetic integer id to
each distinct token string it sees (so the rest of the system, which speaks
token ids, keeps working unchanged).

Capabilities differ by provider, surfaced via ``Capabilities`` so the UI adapts:
- Fireworks: /completions with logprobs (our samplers) AND whole-context via
  ``echo`` (the only cloud path to per-prompt-token logprobs).
- NIM / OpenRouter: chat-only -> generated-token logprobs via /chat/completions
  (no raw continuation, no whole-context echo).
- LM Studio: local OpenAI server with /completions.

Per-request resilience: every HTTP call goes through :meth:`_request`, which
retries ``429 Too Many Requests`` (and a small set of transient 5xx codes) up
to :data:`_MAX_RETRIES` times. The wait time honors the provider's
``Retry-After`` header when present; otherwise we use exponential backoff with
a touch of jitter. This makes both the per-step ``next_distribution`` decode
loop and the native streaming path tolerant of bursty per-account rate limits
(notably the tight serverless RPS Fireworks enforces on freshly released
models like glm-5p2). Custom samplers still fall back to the per-step loop
in :mod:`decoding_sandbox.web.streaming`; that path benefits from retries too.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Iterator
from typing import Any

import httpx

from decoding_sandbox.core.backend import Backend, candidates_from_logprobs
from decoding_sandbox.core.config import ProviderConfig
from decoding_sandbox.core.engine import GenStep
from decoding_sandbox.core.samplers import SamplerDecision
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate

log = logging.getLogger(__name__)

# HTTP statuses we treat as worth one more try. 429 is the headline case
# (per-account RPS throttle on serverless cloud endpoints); the transient
# 5xx group catches "the gateway in front of the model server hiccuped".
# 400/401/403/404 are NOT retried -- they signal a request-shape or auth
# problem that won't fix itself.
_RETRIABLE_STATUSES = frozenset({429, 502, 503, 504})

# Number of retries on top of the initial attempt. 3 means the caller waits
# at most ~7s (1+2+4) of pure backoff before the call fails, which keeps the
# UX snappy while absorbing a brief burst hitting the upstream limit.
_MAX_RETRIES = 3

# Base delay (seconds) for the exponential-backoff fallback when the server
# did not send a Retry-After header. With _MAX_RETRIES=3 the schedule is
# roughly 1s, 2s, 4s, each with up to 250 ms of jitter to desynchronize
# parallel callers (relevant when several browser tabs hit the same backend).
_BASE_BACKOFF_S = 1.0

# Mappable samplers for native server-side generation. The decode loop in
# core.engine treats every step as a fresh ``next_distribution`` call; when
# the user picks one of these standard samplers we instead emit a single
# streaming /completions call with the equivalent server-side params, which
# turns N HTTP requests into 1 SSE response. ``typical`` is not in this set
# because OpenAI-compat doesn't have a typical_p analogue; ``custom``
# obviously can't run remotely either, so both fall back to the per-step
# loop (which still benefits from _request retries).
_NATIVE_SAMPLERS: frozenset[str] = frozenset(
    {"greedy", "temperature", "top_k", "top_p", "min_p"}
)


def _parse_retry_after(value: str | None) -> float | None:
    """Return seconds to wait, or ``None`` when the header is missing/unparsable.

    Fireworks (and most OpenAI-compat servers) return ``Retry-After`` as a
    plain integer-seconds string. RFC 7231 also allows an HTTP-date form;
    we don't bother decoding it here -- the caller will fall back to
    exponential backoff in that case, which is just as good.
    """
    if not value:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return max(0.0, float(s))
    except ValueError:
        return None


class OpenAICompatBackend(Backend):
    def __init__(
        self,
        provider: ProviderConfig,
        model: str | None = None,
        timeout: float = 120.0,
        *,
        max_retries: int = _MAX_RETRIES,
        base_backoff_s: float = _BASE_BACKOFF_S,
        sleep: Any = time.sleep,
    ):
        # ``sleep`` is injectable so tests can drive retry behaviour without
        # actually waiting wall-clock seconds. Default is ``time.sleep`` in
        # production. ``max_retries`` / ``base_backoff_s`` are knobs the
        # ProviderConfig could pipe through later, but defaulting them here
        # keeps the config schema unchanged.
        self.provider = provider
        self.model = model or provider.default_model
        key = provider.api_key() or "not-needed"
        self._client = httpx.Client(
            base_url=provider.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=timeout,
        )
        self._id_to_text: dict[int, str] = {}
        self._text_to_id: dict[str, int] = {}
        self._max_retries = max(0, int(max_retries))
        self._base_backoff_s = max(0.0, float(base_backoff_s))
        self._sleep = sleep

    # -- synthetic token-id space ----------------------------------------- #
    def _intern(self, text: str) -> int:
        if text not in self._text_to_id:
            tid = len(self._text_to_id)
            self._text_to_id[text] = tid
            self._id_to_text[tid] = text
        return self._text_to_id[text]

    def tokenize(self, text: str) -> list[int]:
        # We can't replicate the server tokenizer; treat the text as one unit.
        return [self._intern(text)]

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(self._id_to_text.get(t, "") for t in token_ids)

    def piece(self, token_id: int) -> str:
        return self._id_to_text.get(token_id, "")

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=f"{self.provider.name}:{self.model}",
            full_vocab=False,
            prompt_logprobs=self.provider.supports_prompt_logprobs,
            max_top_logprobs=self.provider.max_top_logprobs,
            can_force_token=self.provider.has_completions,
            notes=(
                "whole-context via echo"
                if self.provider.supports_prompt_logprobs
                else ("raw /completions" if self.provider.has_completions else "chat-only top-k")
            ),
        )

    # -- requests ---------------------------------------------------------- #
    def supports_native_sampler(self, sampler_name: str, sampler_params: dict[str, Any]) -> bool:
        """Can we offload this sampler to the provider's server side?

        ``True`` means a single streaming ``/completions`` call with the
        equivalent ``temperature`` / ``top_p`` / ``top_k`` / ``min_p`` params
        replaces the per-step decode loop -- one HTTP request instead of
        ``max_tokens`` of them, so we stop tripping per-account RPS limits.

        Native streaming requires the provider to expose ``/completions``
        (Fireworks, LM Studio); chat-only paths (NIM, OpenRouter) keep
        running through the per-step loop, which still benefits from the
        retry/backoff added to :meth:`_post`. ``typical_p`` has no
        OpenAI-compat analogue and ``custom`` can't run remotely, so both
        intentionally land on the per-step fallback too.
        """
        del sampler_params  # currently unused; kept for forward-compat
        if not self.provider.has_completions:
            return False
        return sampler_name in _NATIVE_SAMPLERS

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if self.provider.require_parameters:
            body.setdefault("provider", {})["require_parameters"] = True
        r = self._request("post", path, json=body)
        return r.json()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """HTTP call with 429/5xx retry that honors ``Retry-After``.

        This is the single chokepoint for every outbound request: ``_post``
        uses it for JSON bodies, ``_fetch_fireworks_models`` and
        ``fetch_available_models`` use it for the catalogue ``GET``s, and
        the native streaming path uses a sibling :meth:`_stream` so it
        gets the same treatment for the *initial* SSE response code.

        Retry policy:

        - ``2xx`` is returned immediately.
        - ``429`` / ``502`` / ``503`` / ``504`` are retried up to
          ``max_retries`` times. Wait time is the value of the server's
          ``Retry-After`` header (parsed as seconds) when present;
          otherwise exponential backoff (``base_backoff_s * 2**attempt``)
          plus jitter, so concurrent callers don't desynchronize.
        - Every other non-2xx raises via ``raise_for_status`` so the
          caller sees the original ``HTTPStatusError`` (400/401/403/404
          are NEVER retried -- those won't fix themselves and the caller
          needs to see the body for debugging).
        """
        last_response: Any = None
        for attempt in range(self._max_retries + 1):
            response = getattr(self._client, method)(path, **kwargs)
            last_response = response
            status = int(getattr(response, "status_code", 0))
            if 200 <= status < 300:
                return response
            if status in _RETRIABLE_STATUSES and attempt < self._max_retries:
                headers = getattr(response, "headers", None) or {}
                wait = _parse_retry_after(headers.get("Retry-After") if hasattr(headers, "get") else None)
                if wait is None:
                    # Exponential backoff with jitter. Jitter is small (250
                    # ms) so the user-visible delay tracks the "expected"
                    # schedule, but enough to break ties between racing
                    # callers hitting the same upstream bucket.
                    wait = self._base_backoff_s * (2**attempt) + random.uniform(0.0, 0.25)
                log.warning(
                    "%s: %s %s -> HTTP %d; sleeping %.2fs before retry %d/%d",
                    self.provider.name,
                    method.upper(),
                    path,
                    status,
                    wait,
                    attempt + 1,
                    self._max_retries,
                )
                self._sleep(wait)
                continue
            # Either a non-retriable status, or we've exhausted retries.
            response.raise_for_status()
            return response  # pragma: no cover -- raise_for_status raised
        # Loop exited via retry exhaustion -> raise from the last response.
        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError("unreachable: retry loop completed without a response")

    def _cands_from_dict(self, d: dict[str, float]) -> list[TokenCandidate]:
        triples = [(self._intern(tok), tok, float(lp)) for tok, lp in d.items()]
        return candidates_from_logprobs(triples)

    def _cands_from_list(self, items: list[dict]) -> list[TokenCandidate]:
        triples = [(self._intern(i["token"]), i["token"], float(i["logprob"])) for i in items]
        return candidates_from_logprobs(triples)

    def next_distribution(self, token_ids: list[int], top_k: int) -> StepResult:
        text = self.detokenize(token_ids)
        top = max(1, min(top_k, self.provider.max_top_logprobs))
        if self.provider.has_completions:
            data = self._post(
                "/completions",
                {
                    "model": self.model,
                    "prompt": text,
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": top,
                },
            )
            lp = (data.get("choices") or [{}])[0].get("logprobs") or {}
            top_lps = lp.get("top_logprobs") or []
            cands = self._cands_from_dict(top_lps[0]) if top_lps else []
        else:
            data = self._post(
                "/chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": text}],
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": True,
                    "top_logprobs": top,
                },
            )
            content = ((data.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
            cands = self._cands_from_list(content[0].get("top_logprobs", [])) if content else []
        return StepResult(position=len(token_ids), candidates=cands, is_full_vocab=False)

    def score_prompt(
        self, prompt: str, top_k: int, watch_ids: list[int] | None = None
    ) -> list[StepResult]:
        """Whole-context inspection. Uses /completions echo where supported.

        Chat-only providers (NIM, OpenRouter, LM Studio chat) cannot score
        prompt tokens server-side, and cloud tokenization isn't reproducible
        locally, so we refuse rather than silently return an empty list (the
        generic per-prefix fallback in ``Backend.score_prompt`` is meaningless
        here because ``tokenize`` interns the whole text as a single id).
        Callers should branch on ``capabilities.prompt_logprobs`` first.
        """
        if not self.provider.supports_prompt_logprobs:
            raise NotImplementedError(
                f"{self.provider.name!r} has no prompt-logprob support; "
                "this backend cannot do whole-context inspection. Check "
                "capabilities.prompt_logprobs and fall back to "
                "next_distribution() on the prompt instead."
            )

        watch_ids = watch_ids or []
        top = max(1, min(top_k, self.provider.max_top_logprobs))
        data = self._post(
            "/completions",
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": top,
                "echo": True,
            },
        )
        lp = (data.get("choices") or [{}])[0].get("logprobs") or {}
        tokens = lp.get("tokens") or []
        token_lps = lp.get("token_logprobs") or []
        top_lps = lp.get("top_logprobs") or []

        results: list[StepResult] = []
        # Echo returns the prompt tokens; position 0 has no preceding context.
        for i in range(1, len(tokens)):
            cand_dict = top_lps[i] if i < len(top_lps) and top_lps[i] else {}
            cands = self._cands_from_dict(cand_dict)
            actual_text = tokens[i]
            actual_lp = (
                token_lps[i] if i < len(token_lps) and token_lps[i] is not None else float("nan")
            )
            actual_id = self._intern(actual_text)
            chosen = StepResult(0, cands, False).find(actual_id)
            if chosen is None:
                chosen = TokenCandidate(actual_id, actual_text, float(actual_lp), rank=-1)
            step = StepResult(
                position=i,
                candidates=cands,
                is_full_vocab=False,
                chosen=chosen,
                context_text=tokens[i - 1],
            )
            step.watched = {wid: self.lookup_watch(step, wid) for wid in watch_ids}
            results.append(step)
        return results

    # -- native server-side streaming ------------------------------------- #
    def stream_native(
        self,
        prompt: str,
        *,
        sampler_name: str,
        sampler_params: dict[str, Any],
        max_tokens: int,
        top_k: int,
        stop_ids: list[int] | None = None,
    ) -> Iterator[GenStep]:
        """Stream tokens via a single ``/completions`` SSE call.

        This is the antidote to the per-token decode loop in
        :func:`decoding_sandbox.core.engine.generate`: for each of the
        ``_NATIVE_SAMPLERS`` we translate the sampler params to the
        provider's server-side knobs (``temperature`` / ``top_p`` /
        ``top_k`` / ``min_p``) and ask for the *whole* generation in one
        streamed request, with ``logprobs=top_k`` so the per-token
        distribution comes back attached to each chunk.

        The yielded :class:`GenStep`s are wire-compatible with the
        per-step loop so the SSE encoder in
        :mod:`decoding_sandbox.web.streaming` is unchanged: each chunk
        becomes one ``GenStep`` with a populated ``step_result``
        (top-k candidates + chosen) and a synthesized
        ``SamplerDecision`` whose ``kept`` is left empty (server-side
        filtering is opaque to us) but whose ``greedy_token_id`` mirrors
        the highest-logprob entry of the response so the UI still
        flags "changed greedy" cells correctly.

        ``stop_ids`` is honored client-side via the OpenAI ``stop``
        array (we look each id up via ``piece``); ``max_tokens`` is
        enforced by the server. ``finish_reason`` from the *last* chunk
        sets the terminal :class:`GenStep`'s ``stop_reason`` so the
        caller's "stopped on EOS" / "stopped on max_tokens" footer keeps
        working.

        Raises :class:`NotImplementedError` if called on a chat-only
        provider; callers should branch on
        :meth:`supports_native_sampler` first.
        """
        if not self.provider.has_completions:
            raise NotImplementedError(
                f"{self.provider.name!r} has no /completions endpoint; native "
                "streaming requires the raw text-completion path. Fall back "
                "to the per-step decode loop via core.engine.generate."
            )
        top = max(1, min(top_k, self.provider.max_top_logprobs))
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "logprobs": top,
            "stream": True,
        }
        body.update(self._sampler_to_api_params(sampler_name, sampler_params))
        if stop_ids:
            stop_texts: list[str] = []
            for sid in stop_ids:
                txt = self.piece(int(sid))
                if txt:
                    stop_texts.append(txt)
            if stop_texts:
                # OpenAI caps ``stop`` at 4 entries; we honor that to avoid
                # a 400 from picky providers when the caller passed many.
                body["stop"] = stop_texts[:4]
        if self.provider.require_parameters:
            body.setdefault("provider", {})["require_parameters"] = True

        # Compose a short, faithful note so the UI can show what knobs
        # were active server-side. The full sampler_params dict is too
        # noisy for a one-liner.
        note_parts = [f"{sampler_name} (server-side)"]
        for k in ("temperature", "top_p", "top_k", "min_p"):
            if k in body and body[k] is not None:
                note_parts.append(f"{k}={body[k]:g}")
        note = ", ".join(note_parts)

        # Snapshot ids of the prompt for ``tokens_before`` on the first
        # step. Subsequent steps append the emitted token id.
        tokens_before: list[int] = self.tokenize(prompt)
        step_idx = 0
        # Buffer of (text, lp, top_dict_or_list) per emitted token across
        # all chunks; we flush them as GenStep events. SSE chunks can
        # carry 0, 1, or more tokens depending on the provider's
        # streaming granularity.
        pending: list[tuple[str, float, Any]] = []
        last_finish_reason: str | None = None

        for chunk in self._iter_completions_stream(body):
            choices = chunk.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            lp_obj = ch.get("logprobs") or {}
            chunk_tokens = lp_obj.get("tokens") or []
            chunk_lps = lp_obj.get("token_logprobs") or []
            chunk_tops = lp_obj.get("top_logprobs") or []
            for i, tok in enumerate(chunk_tokens):
                lp = (
                    float(chunk_lps[i])
                    if i < len(chunk_lps) and chunk_lps[i] is not None
                    else float("nan")
                )
                top_entry = chunk_tops[i] if i < len(chunk_tops) else None
                pending.append((tok, lp, top_entry))
            fr = ch.get("finish_reason")
            if fr is not None:
                last_finish_reason = str(fr)

        # All chunks consumed: turn the buffer into GenStep events. We
        # delay finishing until here so we know ``finish_reason`` for the
        # terminal step (the provider sends it in the *last* chunk).
        total = len(pending)
        for i, (tok_text, tok_lp, top_entry) in enumerate(pending):
            cands = self._candidates_from_top_entry(top_entry)
            tok_id = self._intern(tok_text)
            chosen = next((c for c in cands if c.token_id == tok_id), None)
            if chosen is None:
                # The emitted token didn't make the top_k cut. Synthesize
                # a candidate with rank=-1 so the UI still has something
                # to show; this matches what ``score_prompt`` does.
                chosen = TokenCandidate(tok_id, tok_text, tok_lp, rank=-1)
            greedy_id = cands[0].token_id if cands else tok_id
            sr = StepResult(
                position=len(tokens_before),
                candidates=cands,
                is_full_vocab=False,
                chosen=chosen,
            )
            is_last = i == total - 1
            stop_reason: str | None = None
            if is_last:
                stop_reason = self._finish_reason_to_stop(last_finish_reason)
            decision = SamplerDecision(
                token_id=tok_id,
                token_text=tok_text,
                kept=[],
                greedy_token_id=greedy_id,
                note=note,
            )
            yield GenStep(
                step=step_idx,
                tokens_before=list(tokens_before),
                step_result=sr,
                decision=decision,
                stop_reason=stop_reason,
            )
            tokens_before.append(tok_id)
            step_idx += 1

    def _iter_completions_stream(self, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield parsed JSON objects from a streaming ``/completions`` call.

        Wraps ``httpx.Client.stream`` so the rest of ``stream_native``
        stays free of SSE-decoding details. ``[DONE]`` terminates the
        stream cleanly; any non-JSON line is logged and skipped (rather
        than crashing the whole stream over a single malformed frame).

        Retry: the *initial* response code goes through the same 429-aware
        path as the rest of the backend, but once the stream is open we
        commit to it -- a mid-stream error becomes an ``HTTPError`` that
        the caller (`stream_generate` in the web layer) turns into a
        terminal ``done`` event with the error string.
        """
        # We can't reuse ``_request`` directly because httpx's streaming
        # API is a context manager, not a plain Response. Replicate the
        # same retry-on-429 logic here, but only for the *opening* of the
        # stream. Once we've started reading bytes, the wire is committed.
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                stream_cm = self._client.stream("POST", "/completions", json=body)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._max_retries:
                    wait = self._base_backoff_s * (2**attempt) + random.uniform(0.0, 0.25)
                    log.warning(
                        "%s: opening stream raised %s; sleeping %.2fs before retry %d/%d",
                        self.provider.name,
                        type(exc).__name__,
                        wait,
                        attempt + 1,
                        self._max_retries,
                    )
                    self._sleep(wait)
                    continue
                raise
            with stream_cm as response:
                status = int(getattr(response, "status_code", 0))
                if status in _RETRIABLE_STATUSES and attempt < self._max_retries:
                    headers = getattr(response, "headers", None) or {}
                    wait = _parse_retry_after(
                        headers.get("Retry-After") if hasattr(headers, "get") else None
                    )
                    if wait is None:
                        wait = self._base_backoff_s * (2**attempt) + random.uniform(0.0, 0.25)
                    log.warning(
                        "%s: POST /completions (stream) -> HTTP %d; sleeping %.2fs before retry %d/%d",
                        self.provider.name,
                        status,
                        wait,
                        attempt + 1,
                        self._max_retries,
                    )
                    self._sleep(wait)
                    continue
                response.raise_for_status()
                for raw in response.iter_lines():
                    line = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError as exc:
                        log.warning(
                            "%s: dropping non-JSON SSE frame: %r (%s)",
                            self.provider.name,
                            payload[:120],
                            exc,
                        )
                return
        # Retry budget exhausted with an exception we couldn't recover from.
        if last_exc is not None:  # pragma: no cover -- defensive
            raise last_exc

    def _candidates_from_top_entry(self, top_entry: Any) -> list[TokenCandidate]:
        """Adapt either /completions (dict) or /chat (list) per-token shapes.

        Fireworks /completions returns ``top_logprobs[i]`` as a dict
        ``{token_text: logprob}``; the chat schema returns a list of
        ``{token, logprob}`` records. Both flow through here so future
        chat-streaming support (NIM, OpenRouter) reuses the parser.
        """
        if top_entry is None:
            return []
        if isinstance(top_entry, dict):
            return self._cands_from_dict(top_entry)
        if isinstance(top_entry, list):
            return self._cands_from_list(top_entry)
        return []

    def _sampler_to_api_params(
        self, name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Translate a built-in sampler to OpenAI-compat request params.

        Only ``_NATIVE_SAMPLERS`` reach this; anything else is rejected
        by :meth:`supports_native_sampler` before we get here. The
        mapping mirrors :mod:`decoding_sandbox.core.samplers` exactly so
        a server-side run produces the same distribution as the
        per-step loop would have. ``temperature`` defaults are pinned
        per-sampler to match ``samplers.BUILTINS``.
        """
        out: dict[str, Any] = {}
        if name == "greedy":
            out["temperature"] = 0
            return out
        if name == "temperature":
            out["temperature"] = float(params.get("temperature", 0.8))
            return out
        if name == "top_k":
            out["temperature"] = float(params.get("temperature", 1.0))
            out["top_k"] = int(params.get("top_k", 40))
            return out
        if name == "top_p":
            out["temperature"] = float(params.get("temperature", 1.0))
            out["top_p"] = float(params.get("top_p", 0.9))
            return out
        if name == "min_p":
            out["temperature"] = float(params.get("temperature", 1.0))
            out["min_p"] = float(params.get("min_p", 0.05))
            return out
        return out  # pragma: no cover -- guarded by supports_native_sampler

    @staticmethod
    def _finish_reason_to_stop(reason: str | None) -> str | None:
        """Map provider ``finish_reason`` to our ``GenStep.stop_reason``.

        ``"length"`` always means we ran out of ``max_tokens``;
        ``"stop"`` means either a stop sequence matched OR the model
        emitted its native EOS. We can't distinguish those two cases
        from the wire (the provider collapses them), so we report
        ``"user_stop"`` -- the engine's CLI footer happens to also say
        "stopped on EOS" only when it knows the EOS id, so this is the
        honest reading.
        """
        if reason == "length":
            return "max_tokens"
        if reason == "stop":
            return "user_stop"
        return None

    # -- model discovery -------------------------------------------------- #
    def fetch_available_models(self, timeout: float = 15.0) -> list[str]:
        """Return the list of model ids this provider currently serves.

        Provider catalogues live at non-uniform paths:

        - NIM, OpenRouter, LM Studio: the OpenAI-compat ``/models`` endpoint
          works as documented and returns ``{"data": [{"id": "..."}, ...]}``.
        - Fireworks: ``/inference/v1/models`` returns 500 (it tries to list
          *deployed* models, not the catalogue). The actual catalogue lives
          at ``/v1/accounts/fireworks/models`` and uses a richer schema; we
          filter to chat-capable serverless models so the picker only shows
          things the OpenAI-compat path can talk to.

        The middleware caches the result so we hit each provider at most
        once per cache TTL (default 6h). Failures bubble up so the caller
        can fall back to the curated static list.
        """
        if self.provider.name == "fireworks":
            return self._fetch_fireworks_models(timeout=timeout)
        # Default OpenAI-compat shape.
        r = self._client.get("/models", timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        models = payload.get("data") if isinstance(payload, dict) else payload
        ids: list[str] = []
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict):
                    mid = m.get("id") or m.get("name")
                    if isinstance(mid, str) and mid:
                        ids.append(mid)
        # Dedupe while preserving order so the first occurrence wins.
        seen: set[str] = set()
        out: list[str] = []
        for mid in ids:
            if mid not in seen:
                seen.add(mid)
                out.append(mid)
        return sorted(out)

    def _fetch_fireworks_models(self, *, timeout: float) -> list[str]:
        """Paginate Fireworks's account-scoped model catalogue.

        Filters to ``HF_BASE_MODEL`` entries with ``supportsServerless=True``
        and no image input, which is the set that actually responds at
        ``POST /inference/v1/chat/completions``.
        """
        base = "https://api.fireworks.ai"  # explicit -- different host than provider.base_url
        url = "/v1/accounts/fireworks/models?pageSize=200"
        out: list[str] = []
        next_token = ""
        # A separate client so the Bearer header reaches the non-compat host.
        with httpx.Client(
            base_url=base,
            headers=dict(self._client.headers),
            timeout=timeout,
        ) as client:
            while True:
                suffix = f"&pageToken={next_token}" if next_token else ""
                r = client.get(url + suffix)
                r.raise_for_status()
                d = r.json()
                for m in d.get("models", []) or []:
                    if not isinstance(m, dict):
                        continue
                    if not m.get("supportsServerless"):
                        continue
                    if m.get("kind") not in ("HF_BASE_MODEL",):
                        continue
                    if m.get("supportsImageInput"):
                        # Vision/multimodal serverless endpoints don't accept the
                        # text-completion ``logprobs`` parameter we rely on.
                        continue
                    name = m.get("name")
                    if isinstance(name, str) and name:
                        out.append(name)
                next_token = d.get("nextPageToken") or ""
                if not next_token:
                    break
        seen: set[str] = set()
        deduped: list[str] = []
        for mid in out:
            if mid not in seen:
                seen.add(mid)
                deduped.append(mid)
        return sorted(deduped)

    def close(self) -> None:
        self._client.close()
