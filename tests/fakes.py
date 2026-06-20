"""Shared test doubles for the decoding sandbox.

A single ``FakeBackend`` covers most needs: pre-seed it with a tokenizer dict
(text -> ids), a piece dict (id -> text), and a distributions dict
(context tuple -> ranked candidate list). The default tokenizer maps each char
to its ord(), which is enough for many tests that don't care about specific
ids but do care that tokenization is *consistent* between calls.
"""

from __future__ import annotations

import math
from typing import Any

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate


class FakeBackend(Backend):
    """In-memory backend used across the test suite."""

    def __init__(
        self,
        *,
        tokens: dict[str, list[int]] | None = None,
        pieces: dict[int, str] | None = None,
        distributions: dict[tuple[int, ...], list[TokenCandidate]] | None = None,
        full_vocab: bool = True,
        prompt_logprobs: bool = True,
        can_force_token: bool = True,
        name: str = "fake",
    ) -> None:
        self.tokens = tokens or {}
        self.pieces = pieces or {}
        self.distributions = distributions or {}
        self.full_vocab = full_vocab
        self.prompt_logprobs = prompt_logprobs
        self.can_force_token_flag = can_force_token
        self.name = name
        self.closed = False

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=self.name,
            full_vocab=self.full_vocab,
            prompt_logprobs=self.prompt_logprobs,
            max_top_logprobs=10,
            can_force_token=self.can_force_token_flag,
        )

    def tokenize(self, text: str) -> list[int]:
        if text in self.tokens:
            return list(self.tokens[text])
        return [ord(ch) for ch in text]

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(self.piece(tid) for tid in token_ids)

    def piece(self, token_id: int) -> str:
        return self.pieces.get(token_id, chr(token_id))

    def next_distribution(self, token_ids: list[int], top_k: int) -> StepResult:
        cands = list(self.distributions.get(tuple(token_ids), []))[:top_k]
        return StepResult(
            position=len(token_ids), candidates=cands, is_full_vocab=self.full_vocab
        )

    def close(self) -> None:
        self.closed = True


def cand(token_id: int, text: str, prob: float, rank: int) -> TokenCandidate:
    """Build a TokenCandidate from a literal probability (converted to logprob)."""
    return TokenCandidate(token_id, text, math.log(prob), rank)


class MockHTTPClient:
    """Replacement for ``httpx.Client`` recording calls and returning canned JSON.

    Usage:
        client = MockHTTPClient({
            ("POST", "/completion"): {"completion_probabilities": [...]},
            ("GET",  "/v1/models"):   {"data": [{"id": "fake-model"}]},
        })
        client.get("/v1/models")           # returns MockResponse with the JSON
        client.post("/completion", json=...) # ditto
    """

    def __init__(
        self,
        routes: dict[tuple[str, str], Any] | None = None,
        *,
        status_overrides: dict[tuple[str, str], int] | None = None,
    ) -> None:
        self.routes = routes or {}
        self.status_overrides = status_overrides or {}
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def _resolve(self, method: str, url: str) -> tuple[int, Any]:
        key = (method, url)
        status = self.status_overrides.get(key, 200)
        if key not in self.routes:
            raise AssertionError(
                f"MockHTTPClient: unregistered request {method} {url!r}. "
                f"Known: {sorted(self.routes)}"
            )
        return status, self.routes[key]

    def get(self, url: str, **kwargs: Any) -> "MockResponse":
        self.calls.append({"method": "GET", "url": url, "kwargs": kwargs})
        status, payload = self._resolve("GET", url)
        return MockResponse(status, payload)

    def post(self, url: str, *, json: Any | None = None, **kwargs: Any) -> "MockResponse":
        self.calls.append({"method": "POST", "url": url, "json": json, "kwargs": kwargs})
        status, payload = self._resolve("POST", url)
        return MockResponse(status, payload)

    def close(self) -> None:
        self.closed = True


class MockResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")
