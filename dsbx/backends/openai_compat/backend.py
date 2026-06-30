"""The :class:`OpenAICompatBackend` class, composed from the package mixins."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import httpx

from dsbx.backends.openai_compat._constants import (
    _BASE_BACKOFF_S,
    _MAX_RETRIES,
    _NATIVE_SAMPLERS,
)
from dsbx.backends.openai_compat._fireworks_ext import _FireworksExtMixin
from dsbx.backends.openai_compat._http import _HttpMixin
from dsbx.backends.openai_compat._parsing import _ParsingMixin
from dsbx.backends.openai_compat._streaming import _StreamingMixin
from dsbx.backends.openai_compat._streaming_echo import _EchoStreamingMixin
from dsbx.backends.openai_compat._tokenizer import _TokenizerMixin
from dsbx.core import usage as usage_mod
from dsbx.core.backend import Backend
from dsbx.core.config import ProviderConfig
from dsbx.core.types import Capabilities, StepResult, TokenCandidate

if TYPE_CHECKING:
    from tokenizers import Tokenizer

log = logging.getLogger(__name__)


class OpenAICompatBackend(
    _StreamingMixin,
    _EchoStreamingMixin,
    _FireworksExtMixin,
    _ParsingMixin,
    _TokenizerMixin,
    _HttpMixin,
    Backend,
):
    def __init__(
        self,
        provider: ProviderConfig,
        model: str | None = None,
        timeout: float = 120.0,
        *,
        max_retries: int = _MAX_RETRIES,
        base_backoff_s: float = _BASE_BACKOFF_S,
        sleep: Any = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ):
        # ``sleep`` is injectable so tests can drive retry behaviour without
        # actually waiting wall-clock seconds. Default is ``time.sleep`` in
        # production. ``max_retries`` / ``base_backoff_s`` are knobs the
        # ProviderConfig could pipe through later, but defaulting them here
        # keeps the config schema unchanged. ``transport`` is the
        # logging hook the web layer installs (see
        # ``dsbx.web.logging.transport``); leaving it
        # ``None`` keeps the CLI path on httpx's default transport.
        self.provider = provider
        self.model = model or provider.default_model
        key = provider.api_key() or "not-needed"
        self._transport = transport
        client_kwargs: dict[str, object] = {
            "base_url": provider.base_url.rstrip("/"),
            "headers": {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            "timeout": timeout,
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.Client(**client_kwargs)  # type: ignore[arg-type]
        self._id_to_text: dict[int, str] = {}
        self._text_to_id: dict[str, int] = {}
        self._max_retries = max(0, int(max_retries))
        self._base_backoff_s = max(0.0, float(base_backoff_s))
        self._sleep = sleep
        # Local HF tokenizer for this specific model. Loaded lazily on the
        # first call to ``tokenize`` / ``detokenize`` / ``piece`` / the
        # capabilities accessor (whichever fires first) via
        # ``hf_hub_download`` against the repo configured in
        # ``provider.tokenizers``. ``_tokenizer_load_attempted`` flips to
        # True after the first attempt (success or graceful failure) so
        # we don't spam the HF Hub or the log on every request when the
        # repo is gated / network is offline. The lock serializes
        # concurrent first-callers within one process. After load:
        # ``_tokenizer`` is the ``tokenizers.Tokenizer`` instance (or
        # None if degraded) and ``_bos_ids`` is populated from its
        # special-token table; both inform the capabilities object.
        self._tokenizer: Tokenizer | None = None
        self._tokenizer_load_attempted: bool = False
        self._tokenizer_load_error: str = ""
        self._tokenizer_load_lock = threading.Lock()
        self._bos_ids: tuple[int, ...] = ()
        # The web layer sets this immediately before invoking a method
        # and clears it after, while holding the per-backend lock from
        # :mod:`dsbx.web.backends`. While it's set, every
        # HTTP call here records itself into the sink (request count +
        # provider-reported token usage). ``None`` means the caller
        # doesn't want accounting -- e.g. CLI ``dsbx generate`` -- so
        # the helpers are no-ops.
        self._active_usage: usage_mod.UsageSink | None = None

    def _provider_flag(self, name: str) -> bool:
        """Resolve a ``supports_*`` flag with per-model override applied.

        Thin wrapper around :meth:`ProviderConfig.flag_for_model` bound
        to ``self.model``. Use this EVERYWHERE the backend reads a
        ``self.provider.supports_*`` flag so a model-specific quirk
        (e.g. Fireworks's ``gpt-oss-20b`` rejecting
        ``sampling_mask: "count"``) automatically reaches the wire
        body, the capabilities envelope, and any consumer that asks
        ``does this model support X?``. Otherwise the override would
        only kick in for ONE call site and silently leak the broken
        flag elsewhere.
        """
        return self.provider.flag_for_model(self.model, name)

    def set_active_usage(self, sink: usage_mod.UsageSink | None) -> None:
        """Bind a usage sink for the duration of the next backend call(s).

        See :class:`dsbx.core.usage.UsageAware`. The per-
        backend lock in the web registry serializes concurrent callers,
        so a plain instance attribute is sufficient -- and avoids the
        :mod:`contextvars` pitfalls around starlette's
        ``iterate_in_threadpool`` (which propagates the calling task's
        context to each worker thread COPY but doesn't propagate worker
        mutations back, breaking any per-stream contextvar set inside
        the body generator).
        """
        self._active_usage = sink

    # -- synthetic token-id space ----------------------------------------- #
    # Offset so synthetic intern ids never collide with real model token
    # ids returned by NewLogProbs (Fireworks vocabs top out around
    # ~256K for the biggest models). Using a clear bit pattern makes
    # synthetic ids visually obvious in debug output too.
    _INTERN_ID_BASE: int = 1 << 24  # 16 777 216

    def _build_prompt(self, prompt: str, prepend_token_ids: Sequence[int] = ()) -> str | list[int]:
        """Return the right ``prompt`` payload for the upstream request.

        When ``prepend_token_ids`` is empty we stay with the historical
        ``prompt: str`` form -- smallest possible request, no extra
        local tokenization step, full backwards-compatibility with the
        path that does not need BOS-conditioning. When it is non-empty
        we switch to TOKEN-ARRAY prompt mode: we tokenize the prompt
        locally with the per-model HF tokenizer (``_ensure_tokenizer``
        is expected to have already succeeded -- the caller validates
        this) and concatenate ``[*prepend_token_ids, *tokenize(prompt)]``
        into a single list of ints. Fireworks (and every other
        OpenAI-compat ``/v1/completions`` we target) accepts this form;
        it's the only way to splice extra ids in FRONT of the prompt
        without going through the user-visible text layer (where
        ``detokenize(BOS) + prompt`` would re-tokenize ambiguously --
        e.g. ``<|begin_of_text|>`` might not round-trip to a single id).

        The contract: ALL three HTTP paths (``score_prompt``,
        ``stream_native``, ``stream_native_with_echo``) route their
        ``"prompt"`` field through this helper, so token-array mode is
        a single-codepoint switch.
        """
        if not prepend_token_ids:
            return prompt
        tok = self._ensure_tokenizer()
        if tok is None:
            # Defensive: the caller is supposed to have validated. We
            # silently fall back to text rather than crash, on the
            # principle that a near-correct request is better than no
            # request -- but we log loudly so this gets fixed.
            log.warning(
                "_build_prompt got prepend_token_ids=%r but no local "
                "tokenizer; dropping the prepended ids and sending "
                "prompt as text. This is a caller bug -- "
                "capabilities.supports_prepend_token_ids was probably "
                "not checked.",
                list(prepend_token_ids),
            )
            return prompt
        body_ids = [int(t) for t in prepend_token_ids]
        body_ids.extend(tok.encode(prompt, add_special_tokens=False).ids)
        return body_ids

    @property
    def capabilities(self) -> Capabilities:
        # Chat-only providers (NIM / OpenRouter -- ``has_completions=false``)
        # are REGISTERED but INERT. The historical
        # ``next_distribution`` chat-completion emulation re-sent the
        # growing ``detokenize(prompt + emitted_so_far)`` as a fresh
        # user message on every step, so the displayed "continuation"
        # was N independent first-responses rather than a real
        # continuation -- misleading data in a sandbox whose whole job
        # is to show the truth. We now raise on that path and mirror
        # that decision into capabilities so the web layer can refuse
        # gracefully and the frontend can disable the option in the
        # backend picker with a tooltip-ready ``notes`` value. The
        # full chat-mode UI (system / user / assistant / new user, no
        # per-step inspection inside an assistant turn) is out of
        # scope here; tracked as a separate PR.
        is_chat_only = not self.provider.has_completions
        # Probe for a local HF tokenizer. Done lazily but eagerly enough
        # that the FIRST capabilities read after construction triggers
        # the load -- the web layer caches the result so the cost is
        # paid once per backend instance. When ``has_local_tokenizer``
        # is true we can (a) splice extra ids in front of the prompt via
        # token-array prompt mode for BOS-conditioning, (b) report a
        # real bos_token_ids so the "fill BOS" helper auto-populates,
        # (c) advertise supports_local_tokenize so the UI shows the live
        # token preview as the user types.
        local_tokenizer = self._ensure_tokenizer() if not is_chat_only else None
        has_local_tokenizer = local_tokenizer is not None
        if is_chat_only:
            notes = "chat-only provider; generation disabled until proper chat-mode UI lands"
        elif self._provider_flag("supports_prompt_logprobs"):
            notes = "whole-context via echo"
        else:
            notes = "raw /completions"
        if not is_chat_only and not has_local_tokenizer:
            mapped = (self.provider.tokenizers or {}).get(self.model)
            if not mapped:
                notes += (
                    f" · no local tokenizer (no [providers."
                    f"{self.provider.name}.tokenizers] entry for "
                    f"{self.model!r}); BOS-conditioning and live token "
                    f"preview disabled"
                )
            elif self._tokenizer_load_error:
                notes += (
                    f" · local tokenizer unavailable "
                    f"({self._tokenizer_load_error}); set HF_TOKEN with "
                    f"access to {mapped!r} to enable BOS-conditioning + "
                    f"live token preview"
                )
        return Capabilities(
            name=f"{self.provider.name}:{self.model}",
            full_vocab=False,
            prompt_logprobs=self._provider_flag("supports_prompt_logprobs"),
            max_top_logprobs=self.provider.max_top_logprobs,
            can_force_token=self.provider.has_completions,
            notes=notes,
            # Provider extension flags surfaced to the UI so the browser
            # can adapt without hard-coding provider names.
            supports_ignore_eos=bool(self._provider_flag("supports_ignore_eos")),
            supports_perf_metrics=bool(self._provider_flag("supports_perf_metrics")),
            supports_service_tier=bool(self._provider_flag("supports_service_tier")),
            supports_sampling_mask=bool(self._provider_flag("supports_sampling_mask")),
            supports_raw_output=bool(self._provider_flag("supports_raw_output")),
            supports_logit_bias=bool(self._provider_flag("supports_logit_bias")),
            supports_combined_echo_stream=bool(
                self._provider_flag("supports_combined_echo_stream")
            ),
            generation_disabled=is_chat_only,
            # Populated from the local HF tokenizer's special-tokens
            # table when available; empty otherwise. Empty means the
            # UI's "fill BOS" helper greys out (and we'd fall back to
            # the user typing the id manually). Token-array prompt mode
            # gates on ``supports_prepend_token_ids``, not on a non-
            # empty BOS list, so the user CAN still type a custom id to
            # condition on even when we don't know the canonical one.
            bos_token_ids=self._bos_ids,
            # Becomes True as soon as we have a working local tokenizer
            # AND the provider has a real /completions endpoint (chat-
            # only paths can't accept token-array prompts because they
            # don't accept ``prompt`` at all). When false the web layer
            # silently drops any incoming ``prepend_token_ids`` and the
            # frontend disables the chip-input.
            supports_prepend_token_ids=(
                has_local_tokenizer and bool(self.provider.has_completions)
            ),
            # Drives the live token preview in the Decode workbench.
            # Independent of ``supports_prepend_token_ids`` so we can
            # potentially light it up for chat-only providers in the
            # future (a chat-mode UI where you see your turn tokenize
            # in real time would be useful even when generation runs
            # through /chat/completions and can't accept token ids).
            supports_local_tokenize=has_local_tokenizer,
        )

    # -- requests ---------------------------------------------------------- #
    def supports_native_sampler(self, sampler_name: str, sampler_params: dict[str, Any]) -> bool:
        """Can we offload this sampler to the provider's server side?

        ``True`` means a single streaming ``/completions`` call with the
        equivalent body params replaces the per-step decode loop -- one
        HTTP request instead of ``max_tokens`` of them, so we stop
        tripping per-account RPS limits.

        Native streaming requires the provider to expose ``/completions``
        (Fireworks, LM Studio); chat-only paths (NIM, OpenRouter) keep
        running through the per-step loop, which still benefits from the
        retry/backoff added to :meth:`_post`. ``typical`` and
        ``mirostat`` are Fireworks-extensions: we only claim native
        support when the provider's config explicitly opts in
        (``supports_typical_p_native`` / ``supports_mirostat``). The
        per-step fallback continues to work either way -- and uses the
        local mirostat-v2 / typical implementations in
        :mod:`dsbx.core.samplers`. ``custom`` can never run
        remotely (no remote code execution).
        """
        del sampler_params  # currently unused; kept for forward-compat
        if not self.provider.has_completions:
            return False
        if sampler_name in _NATIVE_SAMPLERS:
            return True
        if sampler_name == "typical" and self._provider_flag("supports_typical_p_native"):
            return True
        return bool(sampler_name == "mirostat" and self._provider_flag("supports_mirostat"))

    def next_distribution(
        self,
        token_ids: list[int],
        top_k: int,
        *,
        watch_ids: Sequence[int] = (),
    ) -> StepResult:
        text = self.detokenize(token_ids)
        top = max(1, min(top_k, self.provider.max_top_logprobs))
        sampling_mask_count: int | None = None
        if self.provider.has_completions:
            body: dict[str, Any] = {
                "model": self.model,
                "prompt": text,
                "max_tokens": 1,
                "temperature": 0,
            }
            self._attach_logprobs_request(body, top_k=top)
            data = self._post("/completions", body)
            lp = (data.get("choices") or [{}])[0].get("logprobs") or {}
            if self._provider_flag("supports_new_logprobs") and "content" in lp:
                # NewLogProbs (Fireworks): logprobs.content[0].top_logprobs[]
                # carries real token_ids; the per-position
                # ``sampling_mask_count`` rides on the same content entry.
                content = lp.get("content") or []
                if content:
                    entry = content[0]
                    cands = self._cands_from_new_logprobs(entry.get("top_logprobs", []))
                    smc = entry.get("sampling_mask_count")
                    if smc is not None:
                        sampling_mask_count = int(smc)
                else:
                    cands = []
            else:
                # Legacy /completions shape: top_logprobs[i] is a dict.
                top_lps = lp.get("top_logprobs") or []
                cands = self._cands_from_dict(top_lps[0]) if top_lps else []
        else:
            # Chat-only providers (NIM / OpenRouter): we used to silently
            # POST ``[{"role": "user", "content": detokenize(token_ids)}]``
            # to /chat/completions and grab ``top_logprobs[0]`` as if it
            # were the next continuation token. Across the engine's
            # decode loop the user message keeps growing with every
            # picked token, so the model gets a fresh "respond to this
            # slightly-longer user query" prompt each step -- it never
            # sees its own prior emit as assistant, so the displayed
            # "continuation" is actually N independent first-responses,
            # not a real continuation. That's misleading data in a
            # sandbox whose whole job is to expose the truth. Adjacent
            # methods (``score_prompt``, ``stream_native``,
            # ``stream_native_with_echo``) already raise on chat-only
            # providers; we close the inconsistency here. Proper
            # chat-mode UI (system / user / assistant turns, real
            # /chat/completions wire shape) is out of scope -- tracked
            # as a separate PR. Callers should branch on
            # ``capabilities.generation_disabled`` to render the
            # backend as inert instead of catching this exception.
            raise NotImplementedError(
                f"{self.provider.name!r} has no /completions endpoint; "
                "the chat-completion emulation was misleading-by-design "
                "(the per-step decode loop re-sent the growing prompt "
                "as a fresh user message, yielding N independent "
                "first-responses instead of a continuation) and is now "
                "disabled. Use a /completions-capable provider "
                "(fireworks, lmstudio) or a local / remote dsbx backend "
                "until proper chat-mode UI lands."
            )
        # Stamp the per-position mask count onto every candidate at this
        # step so the renderer can read it without a separate plumbing
        # channel. None values are simply ignored downstream.
        if sampling_mask_count is not None:
            for c in cands:
                c.sampling_mask_count = sampling_mask_count
        step = StepResult(position=len(token_ids), candidates=cands, is_full_vocab=False)
        # Top-k-only backend: a watched id either landed in the chunk
        # we already have or it didn't. We do NOT fire a second HTTP
        # request to probe outside top-k (the logit-bias trick the
        # plan mentions is deliberately postponed); ids outside top-k
        # render as ``rank=-1, logprob=NaN`` and the UI shows a dim
        # "—" cell. Same contract :meth:`Backend.lookup_watch`
        # implements generically.
        for wid in watch_ids:
            step.watched[int(wid)] = self.lookup_watch(step, int(wid))
        return step

    def score_prompt(
        self,
        prompt: str,
        top_k: int,
        watch_ids: list[int] | None = None,
        *,
        prepend_token_ids: Sequence[int] = (),
    ) -> list[StepResult]:
        """Whole-context inspection. Uses /completions echo where supported.

        Chat-only providers (NIM, OpenRouter, LM Studio chat) cannot score
        prompt tokens server-side, and cloud tokenization isn't reproducible
        locally, so we refuse rather than silently return an empty list (the
        generic per-prefix fallback in ``Backend.score_prompt`` is meaningless
        here because ``tokenize`` interns the whole text as a single id).
        Callers should branch on ``capabilities.prompt_logprobs`` first.

        Follows the same contract as :meth:`Backend.score_prompt`: for an
        N-token prompt we return N :class:`StepResult`\\s. The first N-1
        rows carry ``chosen`` = the actual prompt token at that position;
        the final row -- the distribution conditioned on the full prompt
        -- has ``chosen=None`` and answers "what does the model predict
        comes next?" (so the inspect UI renders it as ``(predict next)``,
        and ``include_prompt=True`` on the generate path does NOT double
        up that first generated token in the running completion).

        Mechanically we still request ``max_tokens=1`` because that's
        what makes the upstream return ``top_logprobs[N]`` -- the model's
        predicted distribution AT the position after the prompt. The
        provider's "actual next" emission for that slot (which on some
        models, e.g. Fireworks minimax-m2p7, is not even argmax of its
        own ``top_logprobs``) is INTENTIONALLY discarded: there is no
        "actual" prompt token at that position, and labeling the
        provider's continuation as one was misleading.
        """
        if not self._provider_flag("supports_prompt_logprobs"):
            raise NotImplementedError(
                f"{self.provider.name!r} has no prompt-logprob support; "
                "this backend cannot do whole-context inspection. Check "
                "capabilities.prompt_logprobs and fall back to "
                "next_distribution() on the prompt instead."
            )
        if prepend_token_ids and not self._ensure_tokenizer():
            # Caller asked for token-array prompt mode but we have no
            # local tokenizer to build the array with. Hard fail: we
            # CANNOT degrade to ``prompt: str`` here because that would
            # silently drop the prepended ids and the BOS-conditioning
            # the user asked for would never happen. The web layer
            # checks ``capabilities.supports_prepend_token_ids`` first
            # and would normally not reach this branch.
            raise NotImplementedError(
                f"{self.provider.name!r} has no local tokenizer for "
                f"{self.model!r}; prepend_token_ids requires token-array "
                "prompt mode which in turn needs a local replica of the "
                "server tokenizer. Configure [providers."
                f"{self.provider.name}.tokenizers] with the matching HF "
                "repo (and set HF_TOKEN for gated repos), or check "
                "capabilities.supports_prepend_token_ids before calling."
            )

        watch_ids = watch_ids or []
        top = max(1, min(top_k, self.provider.max_top_logprobs))
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": self._build_prompt(prompt, prepend_token_ids),
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
        }
        self._attach_logprobs_request(body, top_k=top)
        data = self._post("/completions", body)
        lp = (data.get("choices") or [{}])[0].get("logprobs") or {}

        if self._provider_flag("supports_new_logprobs") and "content" in lp:
            return self._score_prompt_new_logprobs(lp, watch_ids)
        return self._score_prompt_legacy(lp, watch_ids)

    def _score_prompt_legacy(self, lp: dict[str, Any], watch_ids: list[int]) -> list[StepResult]:
        """Old echo format: ``tokens[]`` + ``token_logprobs[]`` + ``top_logprobs[]``."""
        tokens = lp.get("tokens") or []
        token_lps = lp.get("token_logprobs") or []
        top_lps = lp.get("top_logprobs") or []
        last_idx = len(tokens) - 1
        results: list[StepResult] = []
        for i in range(1, len(tokens)):
            cand_dict = top_lps[i] if i < len(top_lps) and top_lps[i] else {}
            cands = self._cands_from_dict(cand_dict)
            if i < last_idx:
                actual_text = tokens[i]
                actual_lp = (
                    token_lps[i]
                    if i < len(token_lps) and token_lps[i] is not None
                    else float("nan")
                )
                actual_id = self._intern(actual_text)
                chosen = StepResult(0, cands, False).find(actual_id)
                if chosen is None:
                    chosen = TokenCandidate(actual_id, actual_text, float(actual_lp), rank=-1)
            else:
                chosen = None
            step = StepResult(
                position=i,
                candidates=cands,
                is_full_vocab=False,
                chosen=chosen,
                context_text=tokens[i - 1] if i - 1 < len(tokens) else None,
            )
            step.watched = {wid: self.lookup_watch(step, wid) for wid in watch_ids}
            results.append(step)
        return results

    def _score_prompt_new_logprobs(
        self, lp: dict[str, Any], watch_ids: list[int]
    ) -> list[StepResult]:
        """NewLogProbs echo format: ``content[]`` carries per-position entries.

        Each entry: ``{token, token_id, logprob, sampling_mask_count,
        top_logprobs: [...]}``. Trailing entry is the "predict next"
        slot (``chosen=None``); preceding entries are real prompt
        positions whose ``chosen`` IS a top-k member with the correct
        real token_id (no intern fallback).
        """
        content = lp.get("content") or []
        last_idx = len(content) - 1
        results: list[StepResult] = []
        for i in range(1, len(content)):
            entry = content[i] if isinstance(content[i], dict) else {}
            cands = self._cands_from_new_logprobs(entry.get("top_logprobs", []))
            smc_raw = entry.get("sampling_mask_count")
            smc = int(smc_raw) if smc_raw is not None else None
            if smc is not None:
                for c in cands:
                    c.sampling_mask_count = smc
            if i < last_idx:
                actual_text = str(entry.get("token", ""))
                actual_tid_raw = entry.get("token_id")
                actual_id = (
                    int(actual_tid_raw) if actual_tid_raw is not None else self._intern(actual_text)
                )
                if actual_text and actual_id not in self._id_to_text:
                    self._id_to_text[actual_id] = actual_text
                # BOS/EOS and other specials come back as "" -> render the
                # piece so the SEED row shows e.g. ``<|begin_of_sentence|>``
                # instead of the dim ``<empty>``, matching the live preview.
                actual_text = self._surface_text(actual_id, actual_text)
                actual_lp = entry.get("logprob")
                actual_lp_f = float(actual_lp) if actual_lp is not None else float("nan")
                chosen = StepResult(0, cands, False).find(actual_id)
                if chosen is None:
                    chosen = TokenCandidate(
                        actual_id,
                        actual_text,
                        actual_lp_f,
                        rank=-1,
                        sampling_mask_count=smc,
                    )
            else:
                chosen = None
            prev_entry = content[i - 1] if 0 <= (i - 1) < len(content) else {}
            if isinstance(prev_entry, dict):
                prev_tid_raw = prev_entry.get("token_id")
                prev_tid = int(prev_tid_raw) if prev_tid_raw is not None else None
                prev_text = self._surface_text(prev_tid, str(prev_entry.get("token", "")))
            else:
                prev_text = None
            step = StepResult(
                position=i,
                candidates=cands,
                is_full_vocab=False,
                chosen=chosen,
                context_text=prev_text,
            )
            step.watched = {wid: self.lookup_watch(step, wid) for wid in watch_ids}
            results.append(step)
        return results

    # -- native server-side streaming ------------------------------------- #
    def close(self) -> None:
        self._client.close()
