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
import threading
import time
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import httpx

from decoding_sandbox.core import usage as usage_mod
from decoding_sandbox.core.backend import Backend, candidates_from_logprobs
from decoding_sandbox.core.config import ProviderConfig
from decoding_sandbox.core.engine import GenStep
from decoding_sandbox.core.samplers import SamplerDecision
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate

if TYPE_CHECKING:  # avoid the import-time cost of the rust binding for callers
    # that only ever construct a chat-only backend.
    from tokenizers import Tokenizer

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

# Universally mappable samplers (every OpenAI-compat /completions
# endpoint understands these). For ``typical`` and ``mirostat`` we
# additionally check ProviderConfig.supports_typical_p_native /
# supports_mirostat at request time -- those are Fireworks-extensions
# and we don't want to silently ship dead params to providers that
# would either 400 the request or accept-and-ignore the field.
# ``custom`` obviously can't run remotely.
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
        transport: httpx.BaseTransport | None = None,
    ):
        # ``sleep`` is injectable so tests can drive retry behaviour without
        # actually waiting wall-clock seconds. Default is ``time.sleep`` in
        # production. ``max_retries`` / ``base_backoff_s`` are knobs the
        # ProviderConfig could pipe through later, but defaulting them here
        # keeps the config schema unchanged. ``transport`` is the
        # logging hook the web layer installs (see
        # ``decoding_sandbox.web.logging.transport``); leaving it
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
        # :mod:`decoding_sandbox.web.backends`. While it's set, every
        # HTTP call here records itself into the sink (request count +
        # provider-reported token usage). ``None`` means the caller
        # doesn't want accounting -- e.g. CLI ``dsbx generate`` -- so
        # the helpers are no-ops.
        self._active_usage: usage_mod.UsageSink | None = None

    def set_active_usage(self, sink: usage_mod.UsageSink | None) -> None:
        """Bind a usage sink for the duration of the next backend call(s).

        See :class:`decoding_sandbox.core.usage.UsageAware`. The per-
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

    def _intern(self, text: str) -> int:
        if text not in self._text_to_id:
            tid = self._INTERN_ID_BASE + len(self._text_to_id)
            self._text_to_id[text] = tid
            self._id_to_text[tid] = text
        return self._text_to_id[text]

    # -- HF tokenizer ----------------------------------------------------- #
    # Common heuristics for picking the "BOS-ish" id out of a tokenizer
    # that doesn't expose a dedicated ``bos_token`` field. Listed in
    # priority order: we walk this list and take the first matching
    # added/special token. Order matters -- some models reuse
    # ``<|endoftext|>`` as their BOS (Qwen Base), so we have to prefer
    # an explicit start-of-text marker when both are present.
    _BOS_TOKEN_CANDIDATES: tuple[str, ...] = (
        "<|startoftext|>",       # gpt-oss family
        "<|begin_of_text|>",     # Llama 3.x
        "<s>",                   # Llama 2, Mistral
        "<|im_start|>",          # Qwen / ChatML chat marker (best-effort)
        "<|endoftext|>",         # GPT-2, Qwen Base (fallback only)
    )

    def _ensure_tokenizer(self) -> "Tokenizer | None":
        """Lazy-load the HF tokenizer for ``self.model``; cache the result.

        Returns the loaded ``tokenizers.Tokenizer`` instance, or ``None``
        when no tokenizer mapping is configured for this model OR the
        download fails (gated repo without ``HF_TOKEN``, network down,
        404, ...). After the first call the result is cached -- success
        OR failure -- so we never re-attempt the download within the
        lifetime of this backend instance.

        The graceful-failure path is intentional: ``Capabilities`` and
        the basic text-completion calls still work; we just lose the
        token-array prompt mode and the live token preview for this
        particular model. The first-time warning explains why so the
        operator can grant HF access if they want the full UX.
        """
        if self._tokenizer_load_attempted:
            return self._tokenizer
        with self._tokenizer_load_lock:
            if self._tokenizer_load_attempted:
                return self._tokenizer
            try:
                self._tokenizer = self._do_load_tokenizer()
                if self._tokenizer is not None:
                    self._bos_ids = self._discover_bos_ids(self._tokenizer)
            except Exception as exc:  # noqa: BLE001 (we re-raise as warning text)
                self._tokenizer_load_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "tokenizer load failed for %s/%s: %s; "
                    "prepend_token_ids and live token preview will be "
                    "disabled for this model. To enable, "
                    "(a) ensure the mapped repo is correct in "
                    "[providers.%s.tokenizers] and (b) set HF_TOKEN "
                    "in your environment with access to the repo.",
                    self.provider.name,
                    self.model,
                    self._tokenizer_load_error,
                    self.provider.name,
                )
                self._tokenizer = None
            self._tokenizer_load_attempted = True
            return self._tokenizer

    def _do_load_tokenizer(self) -> "Tokenizer | None":
        """Resolve the HF repo for ``self.model`` and load tokenizer.json.

        Returns ``None`` (rather than raising) when no repo is configured
        for this model -- that's a regular "no local tokenizer here"
        outcome, not an error. Real network/gating failures raise and
        get caught + logged in ``_ensure_tokenizer``.
        """
        repo = (self.provider.tokenizers or {}).get(self.model)
        if not repo:
            return None
        # Imports kept local so chat-only / lmstudio paths that never
        # need a tokenizer don't pay the rust-binding import cost.
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        path = hf_hub_download(repo_id=repo, filename="tokenizer.json")
        tok = Tokenizer.from_file(path)
        log.info(
            "loaded HF tokenizer for %s/%s from repo %s (vocab=%d)",
            self.provider.name,
            self.model,
            repo,
            tok.get_vocab_size(),
        )
        return tok

    def _discover_bos_ids(self, tok: "Tokenizer") -> tuple[int, ...]:
        """Best-effort BOS discovery from the tokenizer's special tokens.

        We don't have access to a ``tokenizer_config.json``-style
        explicit ``bos_token`` field via the rust ``tokenizers`` API,
        so we walk the added/special-token decoder and match against a
        small known-suffix list (``_BOS_TOKEN_CANDIDATES``). Returns
        empty when nothing matches -- the UI's "fill BOS" helper will
        grey out and the user can still type any id manually.
        """
        try:
            added = tok.get_added_tokens_decoder()
        except Exception:  # noqa: BLE001
            return ()
        # Build a {content -> id} index over only the SPECIAL added
        # tokens (regular added tokens like merged-word entries don't
        # belong on the "fill BOS" button).
        specials: dict[str, int] = {}
        for tid, tok_obj in added.items():
            if getattr(tok_obj, "special", False):
                specials[tok_obj.content] = int(tid)
        for cand in self._BOS_TOKEN_CANDIDATES:
            if cand in specials:
                return (specials[cand],)
        return ()

    def tokenize(self, text: str) -> list[int]:
        """Tokenize ``text`` locally when a HF tokenizer is configured.

        Falls back to the single-intern-id stub (the historical behaviour
        before per-model tokenizer mapping landed) when no tokenizer is
        available -- e.g. lmstudio (model id is just ``"local-model"``,
        no public HF repo) or a Fireworks model whose ``tokenizer.json``
        we couldn't fetch. The stub still satisfies the callers that
        only need a *handle* per text fragment (e.g. the watch-ids
        text-to-id mapping); only the token-array prompt mode and live
        preview features require the real tokenizer.
        """
        tok = self._ensure_tokenizer()
        if tok is None:
            return [self._intern(text)]
        return list(tok.encode(text, add_special_tokens=False).ids)

    def detokenize(self, token_ids: list[int]) -> str:
        tok = self._ensure_tokenizer()
        if tok is None:
            return "".join(self._id_to_text.get(t, "") for t in token_ids)
        # ``skip_special_tokens=False`` so the BOS / EOS the user
        # explicitly typed in the prepend chip-input round-trip back
        # through the preview as their literal text instead of being
        # silently dropped.
        return tok.decode([int(t) for t in token_ids], skip_special_tokens=False)

    def piece(self, token_id: int) -> str:
        tok = self._ensure_tokenizer()
        if tok is None:
            return self._id_to_text.get(token_id, "")
        # ``id_to_token`` returns the raw vocab string (BPE pieces still
        # carry the GPT-2 ``Ġ`` for word-initial space etc.). Decode of
        # a single id gives the printable surface form, which is what
        # the UI's "piece" RPC consumers expect.
        try:
            return tok.decode([int(token_id)], skip_special_tokens=False)
        except Exception:  # noqa: BLE001
            return self._id_to_text.get(token_id, "")

    def _build_prompt(
        self, prompt: str, prepend_token_ids: Sequence[int] = ()
    ) -> str | list[int]:
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
            notes = (
                "chat-only provider; generation disabled until proper "
                "chat-mode UI lands"
            )
        elif self.provider.supports_prompt_logprobs:
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
            prompt_logprobs=self.provider.supports_prompt_logprobs,
            max_top_logprobs=self.provider.max_top_logprobs,
            can_force_token=self.provider.has_completions,
            notes=notes,
            # Provider extension flags surfaced to the UI so the browser
            # can adapt without hard-coding provider names.
            supports_ignore_eos=bool(self.provider.supports_ignore_eos),
            supports_perf_metrics=bool(self.provider.supports_perf_metrics),
            supports_service_tier=bool(self.provider.supports_service_tier),
            supports_sampling_mask=bool(self.provider.supports_sampling_mask),
            supports_raw_output=bool(self.provider.supports_raw_output),
            supports_logit_bias=bool(self.provider.supports_logit_bias),
            supports_combined_echo_stream=bool(
                self.provider.supports_combined_echo_stream
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
        :mod:`decoding_sandbox.core.samplers`. ``custom`` can never run
        remotely (no remote code execution).
        """
        del sampler_params  # currently unused; kept for forward-compat
        if not self.provider.has_completions:
            return False
        if sampler_name in _NATIVE_SAMPLERS:
            return True
        if sampler_name == "typical" and self.provider.supports_typical_p_native:
            return True
        if sampler_name == "mirostat" and self.provider.supports_mirostat:
            return True
        return False

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if self.provider.require_parameters:
            body.setdefault("provider", {})["require_parameters"] = True
        # Always ask Fireworks for perf_metrics in the response body. The
        # cost is a small object; the benefit is a server-timings panel
        # (TTFT, prefill, generation, speculation acceptance, cached
        # prompt tokens) that turns the educational sandbox into a
        # proper "where is the time going" debugger. Cheap to send; the
        # server ignores the field when unsupported.
        if self.provider.supports_perf_metrics:
            body.setdefault("perf_metrics_in_response", True)
        # Same justification for ``raw_output``: cheap diagnostics on
        # every Fireworks call. The web layer flushes it via a dedicated
        # SSE frame so consumers that don't care just skip it.
        if self.provider.supports_raw_output:
            body.setdefault("raw_output", True)
        r = self._request("post", path, json=body)
        data = r.json()
        # If the provider returned a token-usage block (most do for both
        # /completions and /chat/completions), forward it to the active
        # usage sink so the UI can show real provider-side token counts.
        # Falls through silently when no sink is active or no usage was
        # reported, which is the right default for the test/script paths
        # that don't care.
        u = data.get("usage") if isinstance(data, dict) else None
        if isinstance(u, dict):
            usage_mod.record_tokens(
                self._active_usage,
                prompt_tokens=u.get("prompt_tokens"),
                completion_tokens=u.get("completion_tokens"),
                total_tokens=u.get("total_tokens"),
            )
        perf = data.get("perf_metrics") if isinstance(data, dict) else None
        if isinstance(perf, dict):
            usage_mod.record_perf_metrics(self._active_usage, perf)
        raw_out = data.get("raw_output") if isinstance(data, dict) else None
        if isinstance(raw_out, dict):
            usage_mod.record_raw_output(self._active_usage, raw_out)
        return data

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
            # Count every HTTP attempt (not just successful ones) so the
            # usage sink reflects actual pressure on the provider's RPS
            # budget. A 429-then-200 retry shows up as ``requests=2`` --
            # which is the metric the user wants to see when diagnosing
            # rate-limit issues.
            usage_mod.record_request(self._active_usage)
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

    def _cands_from_new_logprobs(self, items: list[dict]) -> list[TokenCandidate]:
        """Parse a NewLogProbs ``top_logprobs[i]`` array into candidates.

        NewLogProbs (Fireworks) carries real model token ids, the token
        text, logprob, and (for the chosen position only) a
        ``sampling_logprob`` representing the post-filter probability.
        Items are ordered by descending probability per the server, so
        we can take the input list's index as the candidate rank
        without re-sorting.

        Crucially: we use the REAL ``token_id`` instead of calling
        :meth:`_intern`. That's the whole reason we ship
        ``logprobs: true`` -- so ``--watch-id`` references map to the
        same id the model actually emitted, and so the per-token id
        list on each ``GenStep.tokens_before`` is meaningful instead
        of being a stream of synthetic intern hashes. We do still
        populate ``_id_to_text`` for ``piece`` / ``detokenize`` so the
        TUI/web renderer keeps working unchanged.
        """
        out: list[TokenCandidate] = []
        for rank, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            tid_raw = item.get("token_id")
            if tid_raw is None:
                # Some upstreams omit token_id for the runner-ups but
                # include the text. Fall back to intern so the
                # candidate still appears in the table; the warning
                # only fires once per session via the dedup'd cache.
                text = str(item.get("token", ""))
                tid = self._intern(text)
            else:
                tid = int(tid_raw)
                text = str(item.get("token", ""))
                if text:
                    self._id_to_text.setdefault(tid, text)
            lp = item.get("logprob")
            logprob = float(lp) if lp is not None else float("nan")
            out.append(
                TokenCandidate(
                    token_id=tid,
                    text=text,
                    logprob=logprob,
                    rank=rank,
                )
            )
        return out

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
            if self.provider.supports_new_logprobs and "content" in lp:
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

    def _attach_logprobs_request(self, body: dict[str, Any], *, top_k: int) -> None:
        """Add the logprobs-related fields to ``body`` in the right shape.

        Branches on ``provider.supports_new_logprobs``: when on we ship
        ``logprobs: true`` + ``top_logprobs: N`` (NewLogProbs) plus the
        ``sampling_mask: 'count'`` field when ``supports_sampling_mask``;
        otherwise we fall back to the legacy ``logprobs: N`` integer
        form. Centralized here so every callsite (next_distribution,
        score_prompt, stream_native) stays in sync.
        """
        if self.provider.supports_new_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = int(top_k)
            if self.provider.supports_sampling_mask:
                # 'count' asks the server to report how many vocab
                # entries survived the sampler filter at each position;
                # alternative values like 'mask' would return a full
                # boolean tensor which is overkill for the sandbox.
                body["sampling_mask"] = "count"
        else:
            body["logprobs"] = int(top_k)

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
        if not self.provider.supports_prompt_logprobs:
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

        if self.provider.supports_new_logprobs and "content" in lp:
            return self._score_prompt_new_logprobs(lp, watch_ids)
        return self._score_prompt_legacy(lp, watch_ids)

    def _score_prompt_legacy(
        self, lp: dict[str, Any], watch_ids: list[int]
    ) -> list[StepResult]:
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
                    chosen = TokenCandidate(
                        actual_id, actual_text, float(actual_lp), rank=-1
                    )
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
                actual_id = int(actual_tid_raw) if actual_tid_raw is not None else self._intern(
                    actual_text
                )
                if actual_text and actual_id not in self._id_to_text:
                    self._id_to_text[actual_id] = actual_text
                actual_lp = entry.get("logprob")
                actual_lp_f = float(actual_lp) if actual_lp is not None else float("nan")
                chosen = StepResult(0, cands, False).find(actual_id)
                if chosen is None:
                    chosen = TokenCandidate(
                        actual_id, actual_text, actual_lp_f, rank=-1,
                        sampling_mask_count=smc,
                    )
            else:
                chosen = None
            prev_entry = content[i - 1] if 0 <= (i - 1) < len(content) else {}
            prev_text = str(prev_entry.get("token", "")) if isinstance(prev_entry, dict) else None
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
        if prepend_token_ids:
            tok = self._ensure_tokenizer()
            assert tok is not None  # guarded above
            prompt_payload: str | list[int] = [int(t) for t in prepend_token_ids]
            prompt_payload.extend(tok.encode(prompt, add_special_tokens=False).ids)
            if prefix_token_ids:
                prompt_payload.extend(int(t) for t in prefix_token_ids)
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
            if self.provider.supports_ignore_eos:
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
        if self.provider.supports_perf_metrics:
            # Same justification as _post: cheap to request, gives the UI
            # a server-timings panel. Ignored by providers that don't
            # implement it.
            body["perf_metrics_in_response"] = True
        if self.provider.supports_raw_output:
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
        if service_tier and self.provider.supports_service_tier:
            body["service_tier"] = str(service_tier)
        if prompt_cache_key and self.provider.supports_prompt_cache_key:
            body["prompt_cache_key"] = str(prompt_cache_key)
        if logit_bias and self.provider.supports_logit_bias:
            # OpenAI-shaped: {"<token_id>": float in [-100, 100]}.
            # Filter out NaN / out-of-range / non-int keys here rather
            # than at the wire-encoding boundary so we surface a clear
            # error to the user instead of a cryptic 400 from the
            # provider.
            cleaned: dict[str, float] = {}
            for k, v in logit_bias.items():
                try:
                    tid = int(k)
                    bias = float(v)
                except (TypeError, ValueError):
                    continue
                if bias != bias or bias < -100.0 or bias > 100.0:
                    # Skip NaN and out-of-spec values silently; the user
                    # editor in the UI is responsible for catching these
                    # before they hit the wire.
                    continue
                cleaned[str(tid)] = bias
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
        if session_id and self.provider.supports_session_affinity:
            extra_headers["x-session-affinity"] = str(session_id)
            extra_headers["x-multi-turn-session-id"] = str(session_id)

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
        # Buffer of per-token records flushed as GenStep events at the
        # end. Each entry: (token_id_or_None, text, logprob, top_payload,
        # sampling_mask_count). The ``token_id_or_None`` is the REAL
        # model token id when the provider speaks NewLogProbs, else
        # ``None`` (legacy path falls back to ``self._intern(text)``).
        pending: list[tuple[int | None, str, float, Any, int | None]] = []
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
            if self.provider.supports_new_logprobs and "content" in lp_obj:
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
                    pending.append((tid, tok, lp, entry.get("top_logprobs", []), smc))
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
                    pending.append((None, tok, lp, top_entry, None))
            fr = ch.get("finish_reason")
            if fr is not None:
                last_finish_reason = str(fr)

        # All chunks consumed: turn the buffer into GenStep events. We
        # delay finishing until here so we know ``finish_reason`` for the
        # terminal step (the provider sends it in the *last* chunk).
        total = len(pending)
        for i, (tok_id_real, tok_text, tok_lp, top_entry, smc) in enumerate(pending):
            if isinstance(top_entry, list) and self.provider.supports_new_logprobs:
                cands = self._cands_from_new_logprobs(top_entry)
            else:
                cands = self._candidates_from_top_entry(top_entry)
            if tok_id_real is not None:
                tok_id = tok_id_real
                if tok_text and tok_id not in self._id_to_text:
                    self._id_to_text[tok_id] = tok_text
            else:
                tok_id = self._intern(tok_text)
            if smc is not None:
                for c in cands:
                    c.sampling_mask_count = smc
            chosen = next((c for c in cands if c.token_id == tok_id), None)
            if chosen is None:
                # The emitted token didn't make the top_k cut. Synthesize
                # a candidate with rank=-1 so the UI still has something
                # to show; this matches what ``score_prompt`` does.
                chosen = TokenCandidate(
                    tok_id, tok_text, tok_lp, rank=-1, sampling_mask_count=smc
                )
            greedy_id = cands[0].token_id if cands else tok_id
            sr = StepResult(
                position=len(tokens_before),
                candidates=cands,
                is_full_vocab=False,
                chosen=chosen,
            )
            # Per-step watch column: fish each watched id out of the
            # same chunk's top_k. Cloud providers cap top_k at 5
            # (Fireworks) / 20 (NIM/OpenRouter), so ids outside that
            # window render as ``rank=-1, logprob=NaN`` (dim "—" in
            # the UI). The bias-probe trick that would let us read
            # outside-top-k values cheaply is postponed; for now
            # honestly showing "unknown" beats a misleading number.
            for wid in watch_ids:
                sr.watched[int(wid)] = self.lookup_watch(sr, int(wid))
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
        if not self.provider.supports_combined_echo_stream:
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
        if prepend_token_ids:
            tok = self._ensure_tokenizer()
            assert tok is not None  # guarded above
            prompt_payload: str | list[int] = [int(t) for t in prepend_token_ids]
            prompt_payload.extend(tok.encode(prompt, add_special_tokens=False).ids)
            if prefix_token_ids:
                prompt_payload.extend(int(t) for t in prefix_token_ids)
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
        if echo_last is not None and self.provider.supports_echo_last:
            # ``echo_last=N`` (Fireworks) returns logprobs only for the
            # last N tokens of the prompt instead of every position.
            # Saves wire bytes + parsing CPU when the user really only
            # cares about the recent context.
            body["echo_last"] = int(echo_last)
        if not respect_eos:
            if self.provider.supports_ignore_eos:
                body["ignore_eos"] = True
            else:
                usage_mod.add_note(
                    self._active_usage,
                    f"{self.provider.name!r} has no ignore_eos field; "
                    "respect_eos=False has no effect on this backend",
                )
        if self.provider.supports_perf_metrics:
            body["perf_metrics_in_response"] = True
        if self.provider.supports_raw_output:
            body["raw_output"] = True
        if service_tier and self.provider.supports_service_tier:
            body["service_tier"] = str(service_tier)
        if prompt_cache_key and self.provider.supports_prompt_cache_key:
            body["prompt_cache_key"] = str(prompt_cache_key)
        if logit_bias and self.provider.supports_logit_bias:
            cleaned: dict[str, float] = {}
            for k, v in logit_bias.items():
                try:
                    tid = int(k)
                    bias = float(v)
                except (TypeError, ValueError):
                    continue
                if bias != bias or bias < -100.0 or bias > 100.0:
                    continue
                cleaned[str(tid)] = bias
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
        if session_id and self.provider.supports_session_affinity:
            extra_headers["x-session-affinity"] = str(session_id)
            extra_headers["x-multi-turn-session-id"] = str(session_id)

        note_parts = [f"{sampler_name} (server-side, combined echo+stream)"]
        for k in ("temperature", "top_p", "top_k", "min_p"):
            if k in body and body[k] is not None:
                note_parts.append(f"{k}={body[k]:g}")
        note = ", ".join(note_parts)

        # Collect ALL positions first, then split into prompt-echo +
        # emitted-token streams. Fireworks streams ONE position per
        # SSE chunk (regardless of whether it's an echoed prompt
        # token or a freshly generated one), so we can't use
        # "first chunk = all echo, rest = emit" heuristic. Instead
        # we use a three-tier signal, strongest first:
        #
        # 1. ``text_offset`` (NewLogProbs): an echo entry has
        #    ``text_offset < len(prompt)``; the first emit entry
        #    starts at ``text_offset == len(prompt)``. This is the
        #    OpenAI-documented field and Fireworks honors it.
        # 2. ``sampling_mask_count`` / ``sampling_logprob`` presence:
        #    Fireworks omits both for echoed positions (they were
        #    never actually sampled) and populates them for emitted
        #    positions. Acts as a backup when the upstream omits
        #    ``text_offset`` (some providers do).
        # 3. Cumulative ``text`` length: accumulate the token strings
        #    and switch as soon as the running total exceeds
        #    ``len(prompt)``. Used as a last-resort fallback when
        #    neither of the above is available.
        #
        # Entry shape: (token_id_or_None, text, logprob, top_payload,
        # sampling_mask_count, text_offset, has_sampling_signal).
        all_positions: list[
            tuple[int | None, str, float, Any, int | None, int | None, bool]
        ] = []
        last_finish_reason: str | None = None
        for chunk_idx, chunk in enumerate(
            self._iter_completions_stream(body, extra_headers=extra_headers)
        ):
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
            if self.provider.supports_new_logprobs and "content" in lp_obj:
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
                    text_off: int | None = (
                        int(text_off_raw) if text_off_raw is not None else None
                    )
                    # ``sampling_logprob`` exists (even if null) ONLY
                    # for emitted positions on Fireworks; on echo
                    # positions the key is absent entirely.
                    has_signal = (
                        "sampling_logprob" in entry
                        or "sampling_mask_count" in entry
                    )
                    all_positions.append(
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
                    all_positions.append(
                        (None, tok, lp, top_entry, None, text_off, False)
                    )
            fr = ch.get("finish_reason")
            if fr is not None:
                last_finish_reason = str(fr)

        # Split point: walk all positions and classify each as echo or
        # emit using the three-tier signal. The split is the FIRST
        # emit index found; everything before is echo (and we trust
        # the upstream to maintain order, since OpenAI-style completions
        # always echo-then-emit).
        prompt_len = len(prompt)
        split_at = len(all_positions)
        running_text_len = 0
        for idx, (_, tok_text, _, _, _, text_off, has_signal) in enumerate(
            all_positions
        ):
            is_emit = False
            if text_off is not None:
                # Primary signal: text_offset >= len(prompt) means the
                # position starts AT or AFTER the end of the prompt
                # (the first emitted token has text_offset == prompt_len
                # in Fireworks; some providers use > so we accept both).
                is_emit = text_off >= prompt_len
            elif has_signal:
                # Secondary: sampling_logprob/mask_count presence.
                is_emit = True
            else:
                # Tertiary: cumulative text accounting. Once we've
                # echoed at least the full prompt the next position
                # must be the first emit.
                running_text_len += len(tok_text)
                is_emit = running_text_len > prompt_len
            if is_emit:
                split_at = idx
                break

        prompt_records = all_positions[:split_at]
        emit_records = all_positions[split_at:]

        # Build StepResults for prompt-echo positions. ``chosen`` is the
        # token that actually appears at this prompt position; rank
        # comes from looking it up inside the chunk's top_logprobs.
        for pos_idx, (
            tok_id_real,
            tok_text,
            tok_lp,
            top_entry,
            smc,
            _text_off,
            _has_signal,
        ) in enumerate(prompt_records):
            if isinstance(top_entry, list) and self.provider.supports_new_logprobs:
                cands = self._cands_from_new_logprobs(top_entry)
            else:
                cands = self._candidates_from_top_entry(top_entry)
            if tok_id_real is not None:
                tok_id = tok_id_real
                if tok_text and tok_id not in self._id_to_text:
                    self._id_to_text[tok_id] = tok_text
            else:
                tok_id = self._intern(tok_text)
            if smc is not None:
                for c in cands:
                    c.sampling_mask_count = smc
            chosen = next((c for c in cands if c.token_id == tok_id), None)
            if chosen is None:
                # Fireworks (and OpenAI-compat echo in general) emits
                # position 0 with NO ``top_logprobs`` -- the model has no
                # prior context to score against, so the upstream returns
                # ``logprob: 0.0`` as a placeholder rather than a real
                # value. If we propagate that 0.0 the UI renders the
                # token at exp(0.0)=1.0=100%, which is a lie:
                # autoregressive models can't predict position 0 without
                # BOS conditioning. Detect the placeholder by the absence
                # of ``cands`` (an empty top_logprobs list is the
                # upstream's "no data here" signal) and downgrade the
                # logprob to NaN so the UI renders an honest "?" instead
                # of the misleading 100%. Emit positions where the chosen
                # is outside top-K still have a real ``tok_lp`` from the
                # entry's ``logprob`` field and a populated ``cands``, so
                # this guard doesn't touch them.
                effective_lp = float("nan") if not cands else tok_lp
                chosen = TokenCandidate(
                    tok_id, tok_text, effective_lp, rank=-1, sampling_mask_count=smc
                )
            prompt_step = StepResult(
                position=pos_idx,
                candidates=cands,
                is_full_vocab=False,
                chosen=chosen,
                context_text=tok_text,
            )
            # Watch column on echoed prompt positions: same top-k-only
            # contract as the per-emitted-step path. Lets ``include_prompt``
            # runs render the same "P(watched)" rows the inspect path does.
            for wid in watch_ids:
                prompt_step.watched[int(wid)] = self.lookup_watch(prompt_step, int(wid))
            yield prompt_step

        # GenSteps for emitted tokens -- same shape as stream_native.
        # tokens_before starts as the local tokenize() of the prompt
        # (synthetic intern ids); accumulates real emitted ids as we
        # go. The synthetic prompt-id space never overlaps real model
        # ids thanks to ``_INTERN_ID_BASE``.
        tokens_before: list[int] = self.tokenize(prompt)
        step_idx = 0
        total = len(emit_records)
        for i, (
            tok_id_real,
            tok_text,
            tok_lp,
            top_entry,
            smc,
            _text_off,
            _has_signal,
        ) in enumerate(emit_records):
            if isinstance(top_entry, list) and self.provider.supports_new_logprobs:
                cands = self._cands_from_new_logprobs(top_entry)
            else:
                cands = self._candidates_from_top_entry(top_entry)
            if tok_id_real is not None:
                tok_id = tok_id_real
                if tok_text and tok_id not in self._id_to_text:
                    self._id_to_text[tok_id] = tok_text
            else:
                tok_id = self._intern(tok_text)
            if smc is not None:
                for c in cands:
                    c.sampling_mask_count = smc
            chosen = next((c for c in cands if c.token_id == tok_id), None)
            if chosen is None:
                chosen = TokenCandidate(
                    tok_id, tok_text, tok_lp, rank=-1, sampling_mask_count=smc
                )
            greedy_id = cands[0].token_id if cands else tok_id
            sr = StepResult(
                position=len(tokens_before),
                candidates=cands,
                is_full_vocab=False,
                chosen=chosen,
            )
            # Per-step watch column on emitted tokens; same rules as
            # the prompt-echo loop above.
            for wid in watch_ids:
                sr.watched[int(wid)] = self.lookup_watch(sr, int(wid))
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
        stream_kwargs["timeout"] = httpx.Timeout(
            connect=10.0, read=45.0, write=10.0, pool=10.0
        )
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

        Only samplers accepted by :meth:`supports_native_sampler` reach
        this; anything else is rejected before we get here. The mapping
        mirrors :mod:`decoding_sandbox.core.samplers` exactly so a
        server-side run produces the same distribution as the per-step
        loop would have. ``temperature`` defaults are pinned per-sampler
        to match ``samplers.BUILTINS``.

        Penalties (``repetition_penalty`` / ``frequency_penalty`` /
        ``presence_penalty``) ride along on EVERY sampler; we forward
        them only when (a) the value differs from its no-op default
        AND (b) the provider advertises support. ``frequency_penalty``
        and ``presence_penalty`` are part of the standard OpenAI body
        shape so we send them whenever set; ``repetition_penalty`` is
        a Fireworks-specific extension and gated by
        ``provider.supports_repetition_penalty``.
        """
        out: dict[str, Any] = {}
        if name == "greedy":
            out["temperature"] = 0
        elif name == "temperature":
            out["temperature"] = float(params.get("temperature", 0.8))
        elif name == "top_k":
            out["temperature"] = float(params.get("temperature", 1.0))
            out["top_k"] = int(params.get("top_k", 40))
        elif name == "top_p":
            out["temperature"] = float(params.get("temperature", 1.0))
            out["top_p"] = float(params.get("top_p", 0.9))
        elif name == "min_p":
            out["temperature"] = float(params.get("temperature", 1.0))
            out["min_p"] = float(params.get("min_p", 0.05))
        elif name == "typical":
            # Guarded by supports_native_sampler -> supports_typical_p_native.
            out["temperature"] = float(params.get("temperature", 1.0))
            out["typical_p"] = float(params.get("typical_p", 0.95))
        elif name == "mirostat":
            # Guarded by supports_native_sampler -> supports_mirostat.
            # Fireworks uses ``mirostat_target`` (τ, in nats) and
            # ``mirostat_lr`` (η); same names we use locally.
            out["temperature"] = float(params.get("temperature", 1.0))
            out["mirostat_target"] = float(params.get("mirostat_target", 5.0))
            out["mirostat_lr"] = float(params.get("mirostat_lr", 0.1))

        # Penalties. These are sampler-agnostic and flow on every
        # request. The no-op defaults match Sampler / BUILTINS so a
        # user who never touches the inputs sees no penalty fields on
        # the wire (smaller body + zero risk of a strict server
        # 400-ing on an unknown key).
        freq = float(params.get("frequency_penalty", 0.0) or 0.0)
        if freq != 0.0:
            out["frequency_penalty"] = freq
        pres = float(params.get("presence_penalty", 0.0) or 0.0)
        if pres != 0.0:
            out["presence_penalty"] = pres
        rep = float(params.get("repetition_penalty", 1.0) or 1.0)
        if rep != 1.0 and self.provider.supports_repetition_penalty:
            out["repetition_penalty"] = rep
        return out

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
        # We thread the same logging transport through so catalogue fetches
        # show up in the upstream-request log alongside chat/completions calls.
        client_kwargs: dict[str, object] = {
            "base_url": base,
            "headers": dict(self._client.headers),
            "timeout": timeout,
        }
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        with httpx.Client(**client_kwargs) as client:  # type: ignore[arg-type]
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
