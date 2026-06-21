"""Tests for /api/v1/spec/stream and /api/v1/probe.

For spec we use a fake target with ``verify_greedy`` and a small draft. For
probe we monkeypatch the provider_probe runner so we don't actually hit
the network -- the goal is to assert caching and the wire shape, not to
re-test the probe logic.
"""

from __future__ import annotations

import json
import math

import pytest

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate
from tests.fakes import FakeBackend
from tests.web_helpers import build_test_app, make_authed_client


# --------------------------------------------------------------------------- #
# spec/stream
# --------------------------------------------------------------------------- #


class _Target(Backend):
    """Fake target with a built-in ``verify_greedy``.

    Accepts the first ``half`` of every draft, then emits a fixed correction.
    """

    def __init__(self):
        self.calls = 0

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="target", full_vocab=True, prompt_logprobs=True, max_top_logprobs=5
        )

    def tokenize(self, text):
        return [ord(c) for c in text]

    def detokenize(self, ids):
        return "".join(chr(i) for i in ids if 0 < i < 0x110000)

    def piece(self, tid):
        return chr(tid) if 0 < tid < 0x110000 else "?"

    def next_distribution(self, token_ids, top_k):
        return StepResult(
            position=len(token_ids),
            candidates=[TokenCandidate(88, "X", math.log(0.9), 0)],
            is_full_vocab=True,
        )

    def verify_greedy(self, context_ids, draft_ids):
        self.calls += 1
        # Accept first ``len // 2`` drafts; correction is a fixed bonus token.
        n = max(0, len(draft_ids) // 2)
        return n, TokenCandidate(120, "x", math.log(0.5), 0)

    def close(self) -> None:
        pass


class _Draft(Backend):
    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(name="draft", full_vocab=True, prompt_logprobs=True, max_top_logprobs=5)

    def tokenize(self, text):
        return [ord(c) for c in text]

    def detokenize(self, ids):
        return ""

    def piece(self, tid):
        return chr(tid) if 0 < tid < 0x110000 else "?"

    def next_distribution(self, token_ids, top_k):
        # Always proposes a single greedy continuation.
        return StepResult(
            position=len(token_ids),
            candidates=[TokenCandidate(88, "X", math.log(0.8), 0)],
            is_full_vocab=True,
        )

    def close(self) -> None:
        pass


def _parse_sse(body: str) -> list[dict]:
    out: list[dict] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if chunk.startswith("data:"):
            out.append(json.loads(chunk[len("data:") :].strip()))
    return out


def test_spec_stream_emits_rounds_and_done() -> None:
    app = build_test_app({"hf-big": _Target(), "hf-small": _Draft()})
    with make_authed_client(app) as c:
        r = c.post(
            "/api/v1/spec/stream",
            json={
                "target_backend": "hf-big",
                "draft_backend": "hf-small",
                "prompt": "ab",
                "gamma": 4,
                "max_tokens": 6,
            },
        )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    round_events = [e for e in events if e["event"] == "round"]
    assert len(round_events) >= 1
    final = events[-1]
    assert final["event"] == "done"
    assert final.get("error") is None
    assert final["total_proposed"] >= 1
    assert final["total_emitted"] >= 1
    assert isinstance(final["completion"], str)


def test_spec_stream_rejects_same_target_and_draft() -> None:
    app = build_test_app({"hf-big": _Target()})
    with make_authed_client(app) as c:
        r = c.post(
            "/api/v1/spec/stream",
            json={
                "target_backend": "hf-big",
                "draft_backend": "hf-big",
                "prompt": "ab",
                "gamma": 2,
                "max_tokens": 4,
            },
        )
    assert r.status_code == 400


def test_spec_stream_unknown_backend_returns_400() -> None:
    app = build_test_app({"hf-big": _Target()})
    with make_authed_client(app) as c:
        r = c.post(
            "/api/v1/spec/stream",
            json={
                "target_backend": "hf-big",
                "draft_backend": "no-such",
                "prompt": "ab",
                "gamma": 2,
                "max_tokens": 4,
            },
        )
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# /api/v1/probe
# --------------------------------------------------------------------------- #
@pytest.fixture
def probe_app(monkeypatch):
    backend = FakeBackend(
        tokens={"ab": [97, 98]},
        pieces={97: "a", 98: "b"},
        distributions={},
        eos_token_ids=(99,),
    )

    # Stub probe_provider so the test doesn't touch the network. Each call
    # increments a counter so we can assert caching prevents re-runs.
    counter = {"n": 0}

    def fake_probe(prov, model):
        from decoding_sandbox.core.provider_probe import ProbeResult

        counter["n"] += 1
        return ProbeResult(
            provider=prov.name,
            model=prov.default_model,
            chat_logprobs=f"ok (call#{counter['n']})",
            prompt_logprobs="n/a",
        )

    monkeypatch.setattr("decoding_sandbox.core.provider_probe.probe_provider", fake_probe)
    app = build_test_app({"dsbx-host-py": backend})
    return app, counter


def test_probe_returns_rows_and_caches(probe_app) -> None:
    app, counter = probe_app
    with make_authed_client(app) as c:
        r1 = c.get("/api/v1/probe")
        r2 = c.get("/api/v1/probe")
    assert r1.status_code == 200
    data1 = r1.json()
    data2 = r2.json()
    assert data1["fresh"] is True
    assert data2["fresh"] is False  # cache hit on second call
    assert len(data1["rows"]) == 2  # fireworks + nim from default test cfg
    # Calls should not have grown for the second request.
    assert counter["n"] == 2


def test_probe_refresh_bypasses_cache(probe_app) -> None:
    app, counter = probe_app
    with make_authed_client(app) as c:
        c.get("/api/v1/probe")  # warm the cache
        r = c.get("/api/v1/probe?refresh=true")
    assert r.json()["fresh"] is True
    # Each probe row is recomputed -> counter goes up.
    assert counter["n"] == 4
