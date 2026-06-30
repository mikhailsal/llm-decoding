"""Native streaming mixin: the /completions SSE decode loop."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import httpx

from decoding_sandbox.backends.openai_compat._constants import _RETRIABLE_STATUSES
from decoding_sandbox.backends.openai_compat._http import _parse_retry_after
from decoding_sandbox.core import usage as usage_mod
from decoding_sandbox.core.engine import GenStep

if TYPE_CHECKING:
    from tokenizers import Tokenizer

    from decoding_sandbox.core.config import ProviderConfig

log = logging.getLogger(__name__)


class _StreamingMixin:
    # Composite-class attributes / cross-mixin methods set in
    # ``OpenAICompatBackend.__init__`` and the sibling mixins. Declared
    # under TYPE_CHECKING so mypy sees the surface this mixin reaches
    # into without changing runtime behaviour.
    if TYPE_CHECKING:
        provider: ProviderConfig
        model: str
        _client: httpx.Client
        _max_retries: int
        _base_backoff_s: float
        _sleep: Any
        _active_usage: usage_mod.UsageSink | None

        def _provider_flag(self, name: str) -> bool: ...
        def _ensure_tokenizer(self) -> Tokenizer | None: ...
        def _attach_logprobs_request(self, body: dict[str, Any], *, top_k: int) -> None: ...
        def _sampler_to_api_params(self, name: str, params: dict[str, Any]) -> dict[str, Any]: ...
        def _genstep_from_emit_record(
            self,
            record: tuple[int | None, str, float, Any, int | None],
            *,
            tokens_before: list[int],
            step_idx: int,
            is_last: bool,
            last_finish_reason: str | None,
            watch_ids: Sequence[int],
            note: str,
        ) -> GenStep: ...
        def tokenize(self, text: str) -> list[int]: ...
        def detokenize(self, token_ids: list[int]) -> str: ...
        def piece(self, token_id: int) -> str: ...

    def stream_native(
        self,
        prompt: str,
        *,
        sampler_name: str,
        sampler_params: dict[str, Any],
        max_tokens: int,
        top_k: int,
        stop_ids: list[int] | None = None,
        seed: int = 0,
        respect_eos: bool = True,
        service_tier: str | None = None,
        prompt_cache_key: str | None = None,
        session_id: str | None = None,
        logit_bias: dict[int, float] | None = None,
        watch_ids: Sequence[int] = (),
        prefix_token_ids: Sequence[int] = (),
        prepend_token_ids: Sequence[int] = (),
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

        ``seed`` is forwarded as the standard OpenAI ``seed`` field so
        runs that the provider considers reproducible (typically dense
        models on a single replica) actually are. Cloud MoE serving
        notoriously isn't bit-deterministic even with a fixed seed
        because of batch-dependent expert routing; we forward it
        anyway because it removes the sampler-side noise even when the
        kernel-side jitter remains, and the user can see this on the
        ``usage.notes`` channel below.

        ``respect_eos=False`` is honored on providers that opt into
        the Fireworks-style ``ignore_eos`` field (see
        ``provider.supports_ignore_eos``): we ship ``ignore_eos: true``
        and the model keeps emitting tokens past its EOS. On providers
        without that flag (NIM / OpenRouter / LM Studio chat-only) the
        request silently degrades to ``respect_eos=True`` and we tack
        an advisory note onto the active usage sink so the UI can say
        "the cloud ignored this flag" rather than silently lying.

        ``service_tier`` is forwarded when ``provider.supports_service_tier``
        is true (Fireworks: ``priority`` upgrades the request out of the
        shared serverless pool; default ``default``).

        ``prompt_cache_key`` is forwarded when
        ``provider.supports_prompt_cache_key`` is true. Requests sharing
        the same key are routed to the same backend replica to maximize
        KV-cache hit rates -- great for manual decoding where the prompt
        prefix barely changes between steps.

        ``session_id``, when ``provider.supports_session_affinity`` is
        true, becomes two HTTP headers Fireworks recognises:
        ``x-session-affinity`` (sticky routing) and
        ``x-multi-turn-session-id`` (MoE Router Replay / R3 -- the
        expert-routing trace is replayed across turns, making MoE
        generations bit-deterministic across a multi-step session).

        ``stream_options.include_usage`` asks the provider to attach a
        ``usage`` block to the final SSE chunk; we read those numbers
        and feed them to :mod:`decoding_sandbox.core.usage` so the web
        layer can render real prompt/completion token counts even
        though we never tokenized the prompt locally.

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
        if prepend_token_ids and not self._ensure_tokenizer():
            # Same contract as ``score_prompt``: a non-empty
            # ``prepend_token_ids`` requires the local tokenizer so we
            # can build a token-array prompt. Hard fail rather than
            # silently lose the BOS the user asked for.
            raise NotImplementedError(
                f"{self.provider.name!r} has no local tokenizer for "
                f"{self.model!r}; prepend_token_ids requires token-array "
                "prompt mode. Configure [providers."
                f"{self.provider.name}.tokenizers] or check "
                "capabilities.supports_prepend_token_ids first."
            )
        # Build the prompt payload. Two routes:
        # (1) No prepend -> stay in text-prompt mode (smallest body,
        #     full back-compat). ``prefix_token_ids`` (manual-mode user
        #     picks) get detokenized + concatenated as text, same as
        #     before this change landed.
        # (2) With prepend -> switch to TOKEN-ARRAY prompt mode and
        #     concatenate ``[*prepend, *tokenize(prompt),
        #     *prefix_token_ids]``. Doing the manual picks in the same
        #     mode avoids a confusing mixed-representation issue (some
        #     ids as text, some as ints) and is more accurate too --
        #     the upstream parses the exact ids we picked, no
        #     detokenize round-trip ambiguity for edge tokens.
        prompt_payload: str | list[int]
        if prepend_token_ids:
            tok = self._ensure_tokenizer()
            assert tok is not None  # guarded above
            payload_ids: list[int] = [int(t) for t in prepend_token_ids]
            payload_ids.extend(tok.encode(prompt, add_special_tokens=False).ids)
            if prefix_token_ids:
                payload_ids.extend(int(t) for t in prefix_token_ids)
            prompt_payload = payload_ids
        else:
            if prefix_token_ids:
                prompt = prompt + self.detokenize([int(t) for t in prefix_token_ids])
            prompt_payload = prompt
        top = max(1, min(top_k, self.provider.max_top_logprobs))
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt_payload,
            "max_tokens": int(max_tokens),
            "stream": True,
            # ``include_usage`` is a recent OpenAI addition supported by
            # Fireworks and LM Studio; servers that don't recognize it
            # ignore the extra key, so passing it unconditionally is safe.
            "stream_options": {"include_usage": True},
            # ``seed`` is a no-op for providers that don't implement it
            # (their request validator just ignores unknown keys); when
            # they DO implement it, we get reduced (not eliminated)
            # sampling jitter for free.
            "seed": int(seed),
        }
        # Logprobs shape (legacy ``logprobs: N`` vs NewLogProbs
        # ``logprobs: true`` + ``top_logprobs: N`` + ``sampling_mask:
        # 'count'``) is decided by the provider's capabilities.
        self._attach_logprobs_request(body, top_k=top)
        if not respect_eos:
            if self._provider_flag("supports_ignore_eos"):
                # The provider has the Fireworks-style escape hatch:
                # ship ``ignore_eos: true`` and the model keeps emitting
                # tokens past its EOS. The UI's "respect EOS" checkbox
                # is now meaningful instead of being a silent lie.
                body["ignore_eos"] = True
            else:
                # No documented OpenAI-compat field on this provider; the
                # request will still halt on the model's EOS. Surface
                # that to the UI via the usage advisory channel.
                usage_mod.add_note(
                    self._active_usage,
                    f"{self.provider.name!r} has no ignore_eos field; "
                    "respect_eos=False has no effect on this backend",
                )
        if self._provider_flag("supports_perf_metrics"):
            # Same justification as _post: cheap to request, gives the UI
            # a server-timings panel. Ignored by providers that don't
            # implement it.
            body["perf_metrics_in_response"] = True
        if self._provider_flag("supports_raw_output"):
            # Always-on for Fireworks: ``raw_output: true`` makes the
            # provider attach a diagnostics block (prompt_fragments,
            # prompt_token_ids, grammar, ...) describing what the model
            # actually saw vs the text we typed. The "what the model
            # saw" UI panel reads this back via the dedicated
            # ``raw_output`` SSE frame; the cost is "one extra
            # smallish dict per request", well worth it for a learning
            # / debugging sandbox where the answer to "why did it pick
            # this token?" often is "because the chat template ate your
            # system prompt".
            body["raw_output"] = True
        if service_tier and self._provider_flag("supports_service_tier"):
            body["service_tier"] = str(service_tier)
        if prompt_cache_key and self._provider_flag("supports_prompt_cache_key"):
            body["prompt_cache_key"] = str(prompt_cache_key)
        if logit_bias and self._provider_flag("supports_logit_bias"):
            # OpenAI-shaped: {"<token_id>": float in [-100, 100]}.
            # Filter out NaN / out-of-range / non-int keys here rather
            # than at the wire-encoding boundary so we surface a clear
            # error to the user instead of a cryptic 400 from the
            # provider.
            cleaned: dict[str, float] = {}
            for k, v in logit_bias.items():
                try:
                    bias_tid = int(k)
                    bias = float(v)
                except (TypeError, ValueError):
                    continue
                if bias != bias or bias < -100.0 or bias > 100.0:
                    # Skip NaN and out-of-spec values silently; the user
                    # editor in the UI is responsible for catching these
                    # before they hit the wire.
                    continue
                cleaned[str(bias_tid)] = bias
            if cleaned:
                body["logit_bias"] = cleaned
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

        # Build per-request HTTP headers for session affinity / MoE R3.
        # Empty dict short-circuits a no-op down the stack so we don't
        # have to special-case "is there anything to send?" later.
        extra_headers: dict[str, str] = {}
        if session_id and self._provider_flag("supports_session_affinity"):
            extra_headers["x-session-affinity"] = str(session_id)
            extra_headers["x-multi-turn-session-id"] = str(session_id)

        # Compose a short, faithful note so the UI can show what knobs
        # were active server-side. The full sampler_params dict is too
        # noisy for a one-liner.
        note_parts = [f"{sampler_name} (server-side)"]
        for knob in ("temperature", "top_p", "top_k", "min_p"):
            if knob in body and body[knob] is not None:
                note_parts.append(f"{knob}={body[knob]:g}")
        note = ", ".join(note_parts)

        # Snapshot ids of the prompt for ``tokens_before`` on the first
        # step. Subsequent steps append the emitted token id.
        tokens_before: list[int] = self.tokenize(prompt)
        step_idx = 0
        # True incremental streaming with a ONE-token lookahead. Each
        # GenStep is emitted as soon as the *next* token's record lands,
        # so the browser's running-completion view paints token-by-token
        # instead of all at once. We hold back exactly one token because
        # the provider only reveals ``finish_reason`` on the very last
        # chunk -- keeping the final record buffered lets us stamp the
        # terminal ``stop_reason`` onto it without buffering the whole
        # completion (the old "collect everything, then yield" shape was
        # what made streamed tokens appear only when the run finished).
        # Record shape: (token_id_or_None, text, logprob, top_payload,
        # sampling_mask_count). ``token_id_or_None`` is the REAL model id
        # on NewLogProbs providers, else ``None`` (legacy path interns).
        prev_record: tuple[int | None, str, float, Any, int | None] | None = None
        last_finish_reason: str | None = None

        for chunk in self._iter_completions_stream(body, extra_headers=extra_headers):
            # The final SSE chunk in an ``include_usage`` stream has an
            # empty ``choices`` array and a populated ``usage`` block;
            # we handle that case first so the per-token loop below
            # doesn't have to special-case empty chunks.
            u = chunk.get("usage") if isinstance(chunk, dict) else None
            if isinstance(u, dict):
                usage_mod.record_tokens(
                    self._active_usage,
                    prompt_tokens=u.get("prompt_tokens"),
                    completion_tokens=u.get("completion_tokens"),
                    total_tokens=u.get("total_tokens"),
                )
            # ``perf_metrics`` lands in the final chunk under the same
            # name as in non-streaming responses (Fireworks doc says so
            # explicitly). Forward to the usage sink so the web layer
            # can emit a dedicated ``perf`` SSE frame.
            perf = chunk.get("perf_metrics") if isinstance(chunk, dict) else None
            if isinstance(perf, dict):
                usage_mod.record_perf_metrics(self._active_usage, perf)
            # ``raw_output`` is also a final-chunk thing (Fireworks
            # emits it once, alongside the closing usage/perf block).
            # Stash on the sink so the web layer can flush a dedicated
            # ``raw_output`` SSE frame.
            raw_out = chunk.get("raw_output") if isinstance(chunk, dict) else None
            if isinstance(raw_out, dict):
                usage_mod.record_raw_output(self._active_usage, raw_out)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            lp_obj = ch.get("logprobs") or {}
            records: list[tuple[int | None, str, float, Any, int | None]] = []
            if self._provider_flag("supports_new_logprobs") and "content" in lp_obj:
                # NewLogProbs streaming: each chunk carries a content[]
                # of per-position entries with real token_ids and
                # sampling_mask_count. The legacy parallel-arrays
                # path below remains the fallback for providers we
                # haven't explicitly opted in.
                for entry in lp_obj.get("content") or []:
                    if not isinstance(entry, dict):
                        continue
                    tok = str(entry.get("token", ""))
                    tid_raw = entry.get("token_id")
                    tid: int | None = int(tid_raw) if tid_raw is not None else None
                    lp_raw = entry.get("logprob")
                    lp = float(lp_raw) if lp_raw is not None else float("nan")
                    smc_raw = entry.get("sampling_mask_count")
                    smc = int(smc_raw) if smc_raw is not None else None
                    records.append((tid, tok, lp, entry.get("top_logprobs", []), smc))
            else:
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
                    records.append((None, tok, lp, top_entry, None))
            fr = ch.get("finish_reason")
            if fr is not None:
                last_finish_reason = str(fr)
            for rec in records:
                if prev_record is not None:
                    yield self._genstep_from_emit_record(
                        prev_record,
                        tokens_before=tokens_before,
                        step_idx=step_idx,
                        is_last=False,
                        last_finish_reason=last_finish_reason,
                        watch_ids=watch_ids,
                        note=note,
                    )
                    step_idx += 1
                prev_record = rec

        # Flush the final held-back token, now known to be terminal, so it
        # carries the provider's ``finish_reason``-derived ``stop_reason``.
        if prev_record is not None:
            yield self._genstep_from_emit_record(
                prev_record,
                tokens_before=tokens_before,
                step_idx=step_idx,
                is_last=True,
                last_finish_reason=last_finish_reason,
                watch_ids=watch_ids,
                note=note,
            )

    def _iter_completions_stream(
        self,
        body: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield parsed JSON objects from a streaming ``/completions`` call.

        Wraps ``httpx.Client.stream`` so the rest of ``stream_native``
        stays free of SSE-decoding details. ``[DONE]`` terminates the
        stream cleanly; any non-JSON line is logged and skipped (rather
        than crashing the whole stream over a single malformed frame).

        ``extra_headers`` lets the caller attach per-request HTTP headers
        on top of the client defaults (Authorization, Content-Type). The
        primary use today is Fireworks' session-affinity headers
        (``x-session-affinity`` / ``x-multi-turn-session-id``) which enable
        MoE Router Replay for cross-turn determinism.

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
        stream_kwargs: dict[str, Any] = {"json": body}
        if extra_headers:
            stream_kwargs["headers"] = dict(extra_headers)
        # Per-stream timeout overrides the client default so a hung
        # provider (or a TCP black hole) doesn't pin the request for
        # the full ``timeout=120`` window. The ``read`` value is
        # specifically "max gap between SSE frames"; Fireworks emits
        # one frame per token so 45 s is wildly generous in practice
        # but tight enough that a real silence surfaces fast and the
        # web layer's ``stream_generate`` can drain its outer ``with``
        # (closing the connection, RSTing the upstream, and letting
        # the browser's stop button take effect within seconds).
        stream_kwargs["timeout"] = httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=10.0)
        for attempt in range(self._max_retries + 1):
            # Count every attempt to open the stream, matching what
            # ``_request`` does for non-streaming calls. Without this
            # the happy path (one streaming POST, no retry) reported
            # ``requests=0`` to the UI -- making the "0 requests, 20
            # tokens" miracle look like our accounting was broken
            # rather than just an unincremented counter on this path.
            usage_mod.record_request(self._active_usage)
            try:
                stream_cm = self._client.stream("POST", "/completions", **stream_kwargs)
            except Exception as exc:
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
                        "%s: POST /completions (stream) -> HTTP %d; "
                        "sleeping %.2fs before retry %d/%d",
                        self.provider.name,
                        status,
                        wait,
                        attempt + 1,
                        self._max_retries,
                    )
                    self._sleep(wait)
                    continue
                response.raise_for_status()
                try:
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
                except GeneratorExit:
                    # Mirrors the remote-backend cleanup: re-raising
                    # lets ``with stream_cm as response`` close the
                    # httpx response, which closes the socket, which
                    # signals the provider to stop streaming. Without
                    # this, a "stop" click on a happily-streaming
                    # Fireworks generate would still leak the
                    # connection until the response naturally
                    # completed.
                    raise
                return
        # Retry budget exhausted with an exception we couldn't recover from.
        if last_exc is not None:  # pragma: no cover -- defensive
            raise last_exc
