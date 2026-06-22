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
        eos_token_ids: tuple[int, ...] = (),
    ) -> None:
        self.tokens = tokens or {}
        self.pieces = pieces or {}
        self.distributions = distributions or {}
        self.full_vocab = full_vocab
        self.prompt_logprobs = prompt_logprobs
        self.can_force_token_flag = can_force_token
        self.name = name
        self.eos_token_ids = eos_token_ids
        self.closed = False

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=self.name,
            full_vocab=self.full_vocab,
            prompt_logprobs=self.prompt_logprobs,
            max_top_logprobs=10,
            can_force_token=self.can_force_token_flag,
            eos_token_ids=self.eos_token_ids,
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
        return StepResult(position=len(token_ids), candidates=cands, is_full_vocab=self.full_vocab)

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

    Each route's value may be:

    - a plain dict / list / string → returned with status 200 and no headers
      (the legacy shape every existing test uses), OR
    - a list of (status, payload, headers) tuples → each call to that route
      pops the next tuple, so tests can simulate
      ``429 -> 429 -> 200`` retry sequences. The last tuple is replayed if
      the route is called more times than tuples were supplied, which keeps
      "happy path after the retry storm" tests concise.

    ``status_overrides`` is preserved for legacy tests that want to pin a
    single status independent of the route's payload.
    """

    def __init__(
        self,
        routes: dict[tuple[str, str], Any] | None = None,
        *,
        status_overrides: dict[tuple[str, str], int] | None = None,
    ) -> None:
        self.routes: dict[tuple[str, str], Any] = routes or {}
        self.status_overrides = status_overrides or {}
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._route_cursor: dict[tuple[str, str], int] = {}

    def _resolve(self, method: str, url: str) -> "MockResponse":
        key = (method, url)
        if key not in self.routes:
            raise AssertionError(
                f"MockHTTPClient: unregistered request {method} {url!r}. "
                f"Known: {sorted(self.routes)}"
            )
        entry = self.routes[key]
        if isinstance(entry, list) and entry and isinstance(entry[0], tuple):
            idx = min(self._route_cursor.get(key, 0), len(entry) - 1)
            self._route_cursor[key] = idx + 1
            triple = entry[idx]
            # Tuple form: (status, payload[, headers])
            status = int(triple[0])
            payload = triple[1] if len(triple) > 1 else None
            headers = triple[2] if len(triple) > 2 else {}
            return MockResponse(status, payload, headers=headers)
        status = int(self.status_overrides.get(key, 200))
        return MockResponse(status, entry)

    def get(self, url: str, **kwargs: Any) -> "MockResponse":
        self.calls.append({"method": "GET", "url": url, "kwargs": kwargs})
        return self._resolve("GET", url)

    def post(self, url: str, *, json: Any | None = None, **kwargs: Any) -> "MockResponse":
        self.calls.append({"method": "POST", "url": url, "json": json, "kwargs": kwargs})
        return self._resolve("POST", url)

    def close(self) -> None:
        self.closed = True


class MockResponse:
    """Stand-in for ``httpx.Response`` covering the surface our code uses.

    ``headers`` defaults to an empty dict so existing tests (which never
    set it) keep working. The retry path reads ``Retry-After`` off of it
    via ``.get(...)``; new tests opt in to a populated dict.
    """

    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = dict(headers or {})

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")
