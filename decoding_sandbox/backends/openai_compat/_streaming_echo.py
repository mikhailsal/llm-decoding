"""Echo streaming mixin: whole-context (prompt-logprobs) streaming via echo."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from decoding_sandbox.core import usage as usage_mod
from decoding_sandbox.core.engine import GenStep
from decoding_sandbox.core.types import StepResult

if TYPE_CHECKING:
    from tokenizers import Tokenizer

    from decoding_sandbox.core.config import ProviderConfig

log = logging.getLogger(__name__)


class _EchoStreamingMixin:
    # Composite-class attributes / cross-mixin methods set in
    # ``OpenAICompatBackend.__init__`` and sibling mixins. Declared
    # under TYPE_CHECKING so mypy sees the surface this mixin reaches
    # into without touching runtime behaviour.
    if TYPE_CHECKING:
        provider: ProviderConfig
        model: str
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
        def _stepresult_from_echo_record(
            self,
            record: tuple[int | None, str, float, Any, int | None, int | None, bool],
            *,
            pos_idx: int,
            watch_ids: Sequence[int],
        ) -> StepResult: ...
        def _iter_completions_stream(
            self,
            body: dict[str, Any],
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> Iterator[dict[str, Any]]: ...
        def tokenize(self, text: str) -> list[int]: ...
        def detokenize(self, token_ids: list[int]) -> str: ...
        def piece(self, token_id: int) -> str: ...

    def stream_native_with_echo(
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
        echo_last: int | None = None,
        watch_ids: Sequence[int] = (),
        prefix_token_ids: Sequence[int] = (),
        prepend_token_ids: Sequence[int] = (),
    ) -> Iterator[StepResult | GenStep]:
        """Combined ``echo=true`` + ``stream=true`` path -- ONE round trip.

        This is the Phase 5 payoff: when the caller wants both the
        per-prompt-token distribution AND the generated stream (the
        "include prompt logits" mode), the two-request fallback
        (``score_prompt`` + ``stream_native``) becomes a single
        streaming POST.

        The yielded values are heterogeneous on purpose:

        * Items of type :class:`StepResult` -- one per echoed prompt
          position. The caller (typically ``web.streaming``) emits
          these as a ``prompt_score`` frame BEFORE any ``step`` frame
          to preserve the wire order the two-request fallback
          produced.
        * Items of type :class:`GenStep` -- one per emitted token,
          shape-identical to what :meth:`stream_native` yields.

        Switching points are determined entirely by chunk position: the
        Fireworks-documented order is "all echoed positions first, then
        emitted tokens, possibly interleaved across chunks". We rely on
        the fact that the first emitted position's text matches the
        chunk's ``text`` field's first character of the *new*
        continuation (not present in the original prompt).

        ``echo_last`` is forwarded as the provider-specific field; the
        first N echoed positions correspond to the last N tokens of the
        prompt rather than the whole prompt. ``None`` means "echo the
        whole prompt" (standard ``echo=true``).

        Only callable when ``provider.supports_combined_echo_stream`` is
        true; raises :class:`NotImplementedError` otherwise. Callers
        should branch on that capability and fall back to the two-step
        path.
        """
        if not self.provider.has_completions:
            raise NotImplementedError(
                f"{self.provider.name!r} has no /completions endpoint; "
                "stream_native_with_echo requires the raw text-completion path."
            )
        if not self._provider_flag("supports_combined_echo_stream"):
            raise NotImplementedError(
                f"{self.provider.name!r} doesn't advertise "
                "supports_combined_echo_stream; use score_prompt + "
                "stream_native as two separate calls."
            )
        if prepend_token_ids and not self._ensure_tokenizer():
            raise NotImplementedError(
                f"{self.provider.name!r} has no local tokenizer for "
                f"{self.model!r}; prepend_token_ids requires token-array "
                "prompt mode. Configure [providers."
                f"{self.provider.name}.tokenizers] or check "
                "capabilities.supports_prepend_token_ids first."
            )
        # Same two-branch logic as ``stream_native``: token-array mode
        # ONLY when prepend is requested (smallest change, full back-
        # compat); manual picks ride along as either text (default) or
        # token ids (token-array mode). The echoed positions then cover
        # the full ``[prepend, prompt, picks]`` block in one continuous
        # stream of per-position logprobs, which is the whole point of
        # the combined path.
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
            "stream_options": {"include_usage": True},
            "seed": int(seed),
            "echo": True,
        }
        self._attach_logprobs_request(body, top_k=top)
        if echo_last is not None and self._provider_flag("supports_echo_last"):
            # ``echo_last=N`` (Fireworks) returns logprobs only for the
            # last N tokens of the prompt instead of every position.
            # Saves wire bytes + parsing CPU when the user really only
            # cares about the recent context.
            body["echo_last"] = int(echo_last)
        if not respect_eos:
            if self._provider_flag("supports_ignore_eos"):
                body["ignore_eos"] = True
            else:
                usage_mod.add_note(
                    self._active_usage,
                    f"{self.provider.name!r} has no ignore_eos field; "
                    "respect_eos=False has no effect on this backend",
                )
        if self._provider_flag("supports_perf_metrics"):
            body["perf_metrics_in_response"] = True
        if self._provider_flag("supports_raw_output"):
            body["raw_output"] = True
        if service_tier and self._provider_flag("supports_service_tier"):
            body["service_tier"] = str(service_tier)
        if prompt_cache_key and self._provider_flag("supports_prompt_cache_key"):
            body["prompt_cache_key"] = str(prompt_cache_key)
        if logit_bias and self._provider_flag("supports_logit_bias"):
            cleaned: dict[str, float] = {}
            for k, v in logit_bias.items():
                try:
                    bias_tid = int(k)
                    bias = float(v)
                except (TypeError, ValueError):
                    continue
                if bias != bias or bias < -100.0 or bias > 100.0:
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
                body["stop"] = stop_texts[:4]
        if self.provider.require_parameters:
            body.setdefault("provider", {})["require_parameters"] = True
        extra_headers: dict[str, str] = {}
        if session_id and self._provider_flag("supports_session_affinity"):
            extra_headers["x-session-affinity"] = str(session_id)
            extra_headers["x-multi-turn-session-id"] = str(session_id)

        note_parts = [f"{sampler_name} (server-side, combined echo+stream)"]
        for knob in ("temperature", "top_p", "top_k", "min_p"):
            if knob in body and body[knob] is not None:
                note_parts.append(f"{knob}={body[knob]:g}")
        note = ", ".join(note_parts)

        # Incremental echo+stream. The provider sends ONE position per SSE
        # chunk in strict echo-then-emit order, so we classify each
        # position the instant it lands instead of buffering the whole
        # response (the old "collect all, split, then yield" shape made
        # the browser paint generated tokens only after the run closed).
        # Classification mirrors the old batch split exactly -- a
        # three-tier signal, strongest first:
        #
        # 1. ``text_offset`` (NewLogProbs): an echo entry has
        #    ``text_offset < len(prompt)``; the first emit entry starts
        #    at ``text_offset == len(prompt)``. OpenAI-documented and
        #    honored by Fireworks.
        # 2. ``sampling_mask_count`` / ``sampling_logprob`` presence:
        #    omitted for echoed positions, populated for emitted ones.
        #    A backup when the upstream omits ``text_offset``.
        # 3. Cumulative ``text`` length: switch once the running total
        #    exceeds ``len(prompt)``. Last-resort fallback.
        #
        # Echo positions are yielded as ``StepResult`` immediately; the
        # first emit position flips us into emit mode, after which every
        # position is streamed as a ``GenStep`` with the same one-token
        # lookahead ``stream_native`` uses so the terminal step can carry
        # ``stop_reason``. Per-position record shape: (token_id_or_None,
        # text, logprob, top_payload, sampling_mask_count, text_offset,
        # has_sampling_signal).
        prompt_len = len(prompt)
        tokens_before: list[int] = self.tokenize(prompt)
        echo_pos_idx = 0
        emit_step_idx = 0
        running_text_len = 0
        in_emit = False
        prev_emit: tuple[int | None, str, float, Any, int | None, int | None, bool] | None = None
        last_finish_reason: str | None = None
        for chunk in self._iter_completions_stream(body, extra_headers=extra_headers):
            u = chunk.get("usage") if isinstance(chunk, dict) else None
            if isinstance(u, dict):
                usage_mod.record_tokens(
                    self._active_usage,
                    prompt_tokens=u.get("prompt_tokens"),
                    completion_tokens=u.get("completion_tokens"),
                    total_tokens=u.get("total_tokens"),
                )
            perf = chunk.get("perf_metrics") if isinstance(chunk, dict) else None
            if isinstance(perf, dict):
                usage_mod.record_perf_metrics(self._active_usage, perf)
            raw_out = chunk.get("raw_output") if isinstance(chunk, dict) else None
            if isinstance(raw_out, dict):
                usage_mod.record_raw_output(self._active_usage, raw_out)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            lp_obj = ch.get("logprobs") or {}
            positions: list[tuple[int | None, str, float, Any, int | None, int | None, bool]] = []
            if self._provider_flag("supports_new_logprobs") and "content" in lp_obj:
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
                    text_off_raw = entry.get("text_offset")
                    text_off: int | None = int(text_off_raw) if text_off_raw is not None else None
                    # ``sampling_logprob`` exists (even if null) ONLY
                    # for emitted positions on Fireworks; on echo
                    # positions the key is absent entirely.
                    has_signal = "sampling_logprob" in entry or "sampling_mask_count" in entry
                    positions.append(
                        (
                            tid,
                            tok,
                            lp,
                            entry.get("top_logprobs", []),
                            smc,
                            text_off,
                            has_signal,
                        )
                    )
            else:
                chunk_tokens = lp_obj.get("tokens") or []
                chunk_lps = lp_obj.get("token_logprobs") or []
                chunk_tops = lp_obj.get("top_logprobs") or []
                offsets = lp_obj.get("text_offset") or []
                for i, tok in enumerate(chunk_tokens):
                    lp = (
                        float(chunk_lps[i])
                        if i < len(chunk_lps) and chunk_lps[i] is not None
                        else float("nan")
                    )
                    top_entry = chunk_tops[i] if i < len(chunk_tops) else None
                    text_off = (
                        int(offsets[i]) if i < len(offsets) and offsets[i] is not None else None
                    )
                    positions.append((None, tok, lp, top_entry, None, text_off, False))
            fr = ch.get("finish_reason")
            if fr is not None:
                last_finish_reason = str(fr)
            for pos in positions:
                if not in_emit:
                    _tid, pos_text, _lp, _top, _smc, text_off, has_signal = pos
                    if text_off is not None:
                        is_emit = text_off >= prompt_len
                    elif has_signal:
                        is_emit = True
                    else:
                        running_text_len += len(pos_text)
                        is_emit = running_text_len > prompt_len
                    if not is_emit:
                        yield self._stepresult_from_echo_record(
                            pos, pos_idx=echo_pos_idx, watch_ids=watch_ids
                        )
                        echo_pos_idx += 1
                        continue
                    in_emit = True
                # Emit position -- stream with one-token lookahead so the
                # last token can carry the provider's stop_reason.
                if prev_emit is not None:
                    yield self._genstep_from_emit_record(
                        prev_emit[:5],
                        tokens_before=tokens_before,
                        step_idx=emit_step_idx,
                        is_last=False,
                        last_finish_reason=last_finish_reason,
                        watch_ids=watch_ids,
                        note=note,
                    )
                    emit_step_idx += 1
                prev_emit = pos

        # Flush the held-back terminal emit token (if any) with its
        # finish-reason-derived stop_reason.
        if prev_emit is not None:
            yield self._genstep_from_emit_record(
                prev_emit[:5],
                tokens_before=tokens_before,
                step_idx=emit_step_idx,
                is_last=True,
                last_finish_reason=last_finish_reason,
                watch_ids=watch_ids,
                note=note,
            )
