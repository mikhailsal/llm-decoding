"""llama.cpp backend: talks to a running ``llama-server`` over HTTP.

Gives top-k logprobs per position (set ``top_k`` large to approximate the full
distribution). Fast on the P40 with partial GPU offload. Used as the day-to-day
engine and for the Qwen3.5-9B-Base GGUF that the HF path can't host on 6 GB.
"""

from __future__ import annotations

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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._max_top = max_top_logprobs
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._piece_cache: dict[int, str] = {}
        self._model_name = self._fetch_model_name()

    def _fetch_model_name(self) -> str:
        try:
            r = self._client.get("/v1/models")
            r.raise_for_status()
            return r.json()["data"][0]["id"]
        except Exception:  # noqa: BLE001
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

    def next_distribution(self, token_ids: list[int], top_k: int) -> StepResult:
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
        triples = [
            (e["id"], _clean_piece(e.get("token", "")), float(e["logprob"]))
            for e in top
        ]
        cands = candidates_from_logprobs(triples)
        return StepResult(
            position=len(token_ids), candidates=cands, is_full_vocab=False
        )

    def close(self) -> None:
        self._client.close()


def _clean_piece(s: str) -> str:
    # llama.cpp may return a byte-fallback marker for partial UTF-8; keep as-is.
    return s
