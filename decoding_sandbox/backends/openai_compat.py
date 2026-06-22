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
"""

from __future__ import annotations

from typing import Any

import httpx

from decoding_sandbox.core.backend import Backend, candidates_from_logprobs
from decoding_sandbox.core.config import ProviderConfig
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate


class OpenAICompatBackend(Backend):
    def __init__(self, provider: ProviderConfig, model: str | None = None, timeout: float = 120.0):
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
    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if self.provider.require_parameters:
            body.setdefault("provider", {})["require_parameters"] = True
        r = self._client.post(path, json=body)
        r.raise_for_status()
        return r.json()

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
