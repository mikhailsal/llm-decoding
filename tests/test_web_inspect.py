"""Tests for /api/v1/tokenize, /detokenize, /piece, /inspect.

The inspect path mirrors :func:`decoding_sandbox.cli.app.cmd_inspect` but on
the wire: same watch-resolution semantics (text / id / eos), same fallback
to ``next_distribution`` for chat-only providers, same trailing predict-next
row from ``score_prompt``.
"""

from __future__ import annotations

import math

import pytest

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate
from tests.fakes import FakeBackend, cand
from tests.web_helpers import build_test_app, make_authed_client


def _make_inspect_backend() -> FakeBackend:
    return FakeBackend(
        tokens={"ab": [97, 98], " Paris": [200], " London": [201]},
        pieces={97: "a", 98: "b", 88: "X", 89: "Y", 99: "", 200: " Paris", 201: " London"},
        distributions={
            (97,): [
                cand(98, "b", 0.7, 0),
                cand(89, "Y", 0.2, 1),
                cand(200, " Paris", 0.1, 2),
            ],
            (97, 98): [
                cand(88, "X", 0.6, 0),
                cand(89, "Y", 0.25, 1),
                cand(200, " Paris", 0.15, 2),
            ],
        },
        eos_token_ids=(99,),
    )


@pytest.fixture
def client():
    backend = _make_inspect_backend()
    app = build_test_app({"dsbx-host-py": backend})
    with make_authed_client(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# tokenize / detokenize / piece
# --------------------------------------------------------------------------- #
def test_tokenize_returns_ids(client) -> None:
    r = client.post("/api/v1/tokenize", json={"backend": "dsbx-host-py", "text": "ab"})
    assert r.status_code == 200
    assert r.json()["ids"] == [97, 98]


def test_detokenize_roundtrip(client) -> None:
    r = client.post("/api/v1/detokenize", json={"backend": "dsbx-host-py", "ids": [97, 98]})
    assert r.status_code == 200
    assert r.json()["text"] == "ab"


def test_piece_endpoint(client) -> None:
    r = client.post("/api/v1/piece", json={"backend": "dsbx-host-py", "id": 88})
    assert r.status_code == 200
    assert r.json()["text"] == "X"


def test_unknown_backend_returns_404(client) -> None:
    r = client.post("/api/v1/tokenize", json={"backend": "no-such", "text": "ab"})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #
def test_inspect_score_prompt_shape(client) -> None:
    r = client.post(
        "/api/v1/inspect",
        json={"backend": "dsbx-host-py", "prompt": "ab", "top_k": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["is_full_vocab"] is True
    assert data["prompt_logprobs"] is True
    # 2-token prompt -> 2 steps; the last has chosen=None ("predict next").
    steps = data["steps"]
    assert len(steps) == 2
    assert steps[-1]["chosen"] is None
    # is_full_vocab propagates from the backend per-step too.
    assert all(s["is_full_vocab"] for s in steps)


def test_inspect_watch_text_resolves_to_single_token(client) -> None:
    r = client.post(
        "/api/v1/inspect",
        json={
            "backend": "dsbx-host-py",
            "prompt": "ab",
            "top_k": 5,
            "watch_texts": [" Paris"],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["watches"]) == 1
    w = data["watches"][0]
    assert w["source"] == "text"
    assert w["token_id"] == 200
    assert w["piece"] == " Paris"
    # The watch column lands in every step's "watched" map.
    for step in data["steps"]:
        watched_ids = {entry["token_id"] for entry in step["watched"]}
        assert 200 in watched_ids


def test_inspect_watch_id_and_eos(client) -> None:
    r = client.post(
        "/api/v1/inspect",
        json={
            "backend": "dsbx-host-py",
            "prompt": "ab",
            "watch_ids": [89],
            "watch_eos": True,
        },
    )
    data = r.json()
    sources = {w["source"] for w in data["watches"]}
    assert {"id", "eos"} <= sources
    # EOS id from fake backend is 99 (Capabilities.eos_token_ids).
    eos = next(w for w in data["watches"] if w["source"] == "eos")
    assert eos["token_id"] == 99


def test_inspect_watch_dedupes_across_sources(client) -> None:
    """The same token id arriving from --watch and --watch-id should appear
    once in the response, with the first label winning."""
    r = client.post(
        "/api/v1/inspect",
        json={
            "backend": "dsbx-host-py",
            "prompt": "ab",
            "watch_texts": [" Paris"],
            "watch_ids": [200],
        },
    )
    data = r.json()
    assert len(data["watches"]) == 1
    assert data["watches"][0]["source"] == "text"


def test_inspect_chat_only_backend_uses_next_token_path() -> None:
    """OpenAICompat-style backends without prompt logprobs fall back to a
    single next-token distribution row rather than re-scoring per prefix."""

    class _ChatOnly(Backend):
        @property
        def capabilities(self) -> Capabilities:
            return Capabilities(
                name="chat-only",
                full_vocab=False,
                prompt_logprobs=False,
                max_top_logprobs=5,
            )

        def tokenize(self, text):
            return [ord(c) for c in text]

        def detokenize(self, ids):
            return "".join(chr(i) for i in ids)

        def piece(self, tid):
            return chr(tid)

        def next_distribution(self, token_ids, top_k):
            return StepResult(
                position=len(token_ids),
                candidates=[TokenCandidate(88, "X", math.log(0.9), 0)],
                is_full_vocab=False,
            )

    # The web app's chat-only detection asks for ``OpenAICompatBackend`` by
    # class name. Use that exact class name so the production path matches.
    _ChatOnly.__name__ = "OpenAICompatBackend"
    app = build_test_app({"chat": _ChatOnly()})
    with make_authed_client(app) as c:
        r = c.post(
            "/api/v1/inspect",
            json={"backend": "chat", "prompt": "ab", "top_k": 5},
        )
    assert r.status_code == 200
    data = r.json()
    assert len(data["steps"]) == 1
    assert "next-token" in data["note"]
    assert data["prompt_logprobs"] is False
