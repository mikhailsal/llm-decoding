"""Fireworks-extension mixin: logprob request shaping, sampler mapping, catalogue."""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


class _FireworksExtMixin:
    def _attach_logprobs_request(self, body: dict[str, Any], *, top_k: int) -> None:
        """Add the logprobs-related fields to ``body`` in the right shape.

        Branches on ``provider.supports_new_logprobs``: when on we ship
        ``logprobs: true`` + ``top_logprobs: N`` (NewLogProbs) plus the
        ``sampling_mask: 'count'`` field when ``supports_sampling_mask``;
        otherwise we fall back to the legacy ``logprobs: N`` integer
        form. Centralized here so every callsite (next_distribution,
        score_prompt, stream_native) stays in sync.
        """
        if self._provider_flag("supports_new_logprobs"):
            body["logprobs"] = True
            body["top_logprobs"] = int(top_k)
            if self._provider_flag("supports_sampling_mask"):
                # 'count' asks the server to report how many vocab
                # entries survived the sampler filter at each position;
                # alternative values like 'mask' would return a full
                # boolean tensor which is overkill for the sandbox.
                body["sampling_mask"] = "count"
        else:
            body["logprobs"] = int(top_k)

    def _sampler_to_api_params(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
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
        if rep != 1.0 and self._provider_flag("supports_repetition_penalty"):
            out["repetition_penalty"] = rep
        return out

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
                exclude = set(self.provider.exclude_models or ())
                for m in d.get("models", []) or []:
                    if not isinstance(m, dict):
                        continue
                    if not m.get("supportsServerless"):
                        continue
                    # Generative LLMs only. ``HF_BASE_MODEL`` is the bulk;
                    # ``CUSTOM_MODEL`` covers provider-tuned variants (e.g.
                    # Fireworks's ``qwen3p*-plus``). We deliberately EXCLUDE
                    # ``EMBEDDING_MODEL`` (not generative) and
                    # ``FLUMINA_BASE_MODEL`` (image generation).
                    if m.get("kind") not in ("HF_BASE_MODEL", "CUSTOM_MODEL"):
                        continue
                    # NOTE: we used to drop ``supportsImageInput`` models on the
                    # theory that vision endpoints reject text ``/completions``
                    # + logprobs. That was WRONG -- the Kimi family and the
                    # Qwen-plus models are multimodal yet answer the text
                    # completions path fine (verified by probe), so the filter
                    # silently hid half the catalogue. Models that genuinely
                    # only speak /chat/completions (e.g. ``minimax-m3``, which
                    # HANGS on /completions) are handled by the explicit
                    # ``exclude_models`` denylist instead of a fragile flag.
                    name = m.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    if name in exclude:
                        continue
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
