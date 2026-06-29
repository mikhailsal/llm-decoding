"""llama.cpp backend: talks to a running ``llama-server`` over HTTP.

Gives top-k logprobs per position (set ``top_k`` large to approximate the full
distribution). Fast on the GPU with partial GPU offload. Used as the day-to-day
engine and for the Qwen3.5-9B-Base GGUF that the HF path can't host on 6 GB.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from decoding_sandbox.core.backend import Backend, candidates_from_logprobs
from decoding_sandbox.core.types import Capabilities, StepResult


class LlamaCppBackend(Backend):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        max_top_logprobs: int = 40,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._max_top = max_top_logprobs
        # ``transport`` is optional so the CLI path keeps using httpx's
        # default transport (no SQLAlchemy / aiosqlite dependency). The
        # web middleware injects a ``LoggingTransport`` here so every
        # /tokenize / /completion call lands in the upstream-request log.
        client_kwargs: dict[str, object] = {"base_url": self.base_url, "timeout": timeout}
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.Client(**client_kwargs)  # type: ignore[arg-type]
        self._piece_cache: dict[int, str] = {}
        self._model_name = self._fetch_model_name()

    def _fetch_model_name(self) -> str:
        try:
            r = self._client.get("/v1/models")
            r.raise_for_status()
            return r.json()["data"][0]["id"]
        except Exception:
            return "llama.cpp"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=f"llamacpp:{self._model_name}",
            full_vocab=False,
            prompt_logprobs=False,  # derived per-prefix (O(n) calls), not native
            max_top_logprobs=self._max_top,
            can_force_token=True,
            notes="top-k logprobs via /completion n_probs; whole-context is re-derived per prefix.",
        )

    def tokenize(self, text: str) -> list[int]:
        r = self._client.post("/tokenize", json={"content": text})
        r.raise_for_status()
        toks = r.json()["tokens"]
        # Some builds return [{"id":..}], others [int]; normalize to ints.
        return [t["id"] if isinstance(t, dict) else int(t) for t in toks]

    def detokenize(self, token_ids: list[int]) -> str:
        r = self._client.post("/detokenize", json={"tokens": token_ids})
        r.raise_for_status()
        return r.json().get("content", "")

    def piece(self, token_id: int) -> str:
        if token_id not in self._piece_cache:
            self._piece_cache[token_id] = self.detokenize([token_id])
        return self._piece_cache[token_id]

    def next_distribution(
        self,
        token_ids: list[int],
        top_k: int,
        *,
        watch_ids: Sequence[int] = (),
    ) -> StepResult:
        n_probs = max(1, min(top_k, self._max_top))
        body = {
            "prompt": token_ids,
            "n_predict": 1,
            "n_probs": n_probs,
            "temperature": 0.0,
            "cache_prompt": True,
            "post_sampling_probs": False,
        }
        r = self._client.post("/completion", json=body)
        r.raise_for_status()
        data = r.json()
        cp = data.get("completion_probabilities") or []
        if not cp:
            return StepResult(position=len(token_ids), candidates=[], is_full_vocab=False)
        top = cp[0].get("top_logprobs", [])
        triples = [(e["id"], _clean_piece(e.get("token", "")), float(e["logprob"])) for e in top]
        cands = candidates_from_logprobs(triples)
        step = StepResult(position=len(token_ids), candidates=cands, is_full_vocab=False)
        # Top-k-only HTTP backend: no way to fish out an outside-top-k
        # logprob from this endpoint, so we fall back to
        # ``lookup_watch`` (real candidate when in top-k, rank=-1/NaN
        # otherwise). Bumping ``top_k`` on the request is the user's
        # escape hatch.
        for wid in watch_ids:
            step.watched[int(wid)] = self.lookup_watch(step, int(wid))
        return step

    def close(self) -> None:
        self._client.close()


def _clean_piece(s: str) -> str:
    # llama.cpp may return a byte-fallback marker for partial UTF-8; keep as-is.
    return s
