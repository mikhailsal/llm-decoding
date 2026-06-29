"""Server endpoint tests using FastAPI's TestClient and a FakeBackend.

The server is built around the existing ``Backend`` protocol, so the
test-suite ``FakeBackend`` is enough to exercise every route -- no real
model load, no CUDA, no httpx wire. SSE generate is tested by tearing
apart the ``text/event-stream`` body into per-step JSON payloads and
asserting on the parsed structure (including the terminating ``done``
event).
"""

from __future__ import annotations

import json
import math
import threading
import time

import pytest
from fastapi.testclient import TestClient

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.types import StepResult, TokenCandidate
from decoding_sandbox.server import schemas as S
from decoding_sandbox.server.app import BackendSlot, make_app
from tests.fakes import FakeBackend, cand


def _make_fake(**overrides) -> FakeBackend:
    """A FakeBackend with two deterministic next-token distributions.

    Tokens 'a' (id 97) and 'b' (id 98) are recognized; from any context
    the next step ranks 'X' over 'Y' with 50/50 probs. Used to exercise
    next_distribution / score_prompt / generate without any actual model
    plumbing.
    """
    return FakeBackend(
        tokens=overrides.pop("tokens", {"ab": [97, 98]}),
        pieces=overrides.pop("pieces", {97: "a", 98: "b", 88: "X", 89: "Y"}),
        distributions=overrides.pop(
            "distributions",
            {
                (97,): [cand(98, "b", 0.6, 0), cand(89, "Y", 0.4, 1)],
                (97, 98): [cand(88, "X", 0.6, 0), cand(89, "Y", 0.4, 1)],
                (97, 98, 88): [cand(88, "X", 0.55, 0), cand(89, "Y", 0.45, 1)],
                (97, 98, 88, 88): [cand(88, "X", 0.55, 0), cand(89, "Y", 0.45, 1)],
                (97, 98, 88, 88, 88): [cand(88, "X", 0.55, 0), cand(89, "Y", 0.45, 1)],
            },
        ),
        eos_token_ids=overrides.pop("eos_token_ids", (99,)),
        **overrides,
    )


@pytest.fixture
def client():
    """One TestClient per test, wrapping a fresh FakeBackend."""
    backend = _make_fake()
    app = make_app(backend, backend_kind="fake-kind")
    with TestClient(app) as c:
        c._backend = backend  # tiny back-channel so tests can mutate the backend
        yield c


# --------------------------------------------------------------------------- #
# /v1/info
# --------------------------------------------------------------------------- #
def test_info_returns_capabilities_and_backend_kind(client) -> None:
    r = client.get("/v1/info")
    assert r.status_code == 200
    data = r.json()
    assert data["backend_kind"] == "fake-kind"
    caps = data["capabilities"]
    assert caps["name"] == "fake"
    assert caps["full_vocab"] is True
    assert caps["max_top_logprobs"] == 10
    assert caps["eos_token_ids"] == [99]
    # engine_version should be a string (we don't pin it -- just non-empty).
    assert isinstance(data["engine_version"], str) and data["engine_version"]


def test_info_exposes_bos_token_ids_and_prepend_support() -> None:
    """``bos_token_ids`` and ``supports_prepend_token_ids`` flow through /info.

    The frontend's "fill BOS" helper reads these two fields out of the
    same payload it uses to decide which sampling knobs to show. We pin
    the wire shape here so a future refactor of Capabilities can't
    silently drop the fields and quietly grey out the helper for
    everyone -- the kind of regression that only surfaces when a user
    notices the button no longer works on their model.
    """
    backend = _make_fake(bos_token_ids=(7, 8), supports_prepend_token_ids=True)
    app = make_app(backend, backend_kind="fake-kind")
    with TestClient(app) as c:
        caps = c.get("/v1/info").json()["capabilities"]
    assert caps["bos_token_ids"] == [7, 8]
    assert caps["supports_prepend_token_ids"] is True


def test_info_loaded_model_is_optional() -> None:
    """When the wrapped backend exposes no model attribute, the server
    reports null rather than guessing."""
    backend = _make_fake()
    # FakeBackend doesn't define ``loaded_model`` / ``model_path`` / ``model``,
    # so the server's sniffer should return None.
    app = make_app(backend, backend_kind="fake-kind")
    with TestClient(app) as c:
        assert c.get("/v1/info").json()["loaded_model"] is None


def test_info_loaded_model_picked_up_from_backend_attributes() -> None:
    backend = _make_fake()
    backend.model_path = "/tmp/some.gguf"
    app = make_app(backend, backend_kind="llamacpp-py")
    with TestClient(app) as c:
        assert c.get("/v1/info").json()["loaded_model"] == "/tmp/some.gguf"


# --------------------------------------------------------------------------- #
# Tokenization endpoints
# --------------------------------------------------------------------------- #
def test_tokenize_returns_ids(client) -> None:
    r = client.post("/v1/tokenize", json={"text": "ab"})
    assert r.status_code == 200
    assert r.json() == {"ids": [97, 98]}


def test_detokenize_round_trip(client) -> None:
    r = client.post("/v1/detokenize", json={"ids": [97, 98]})
    assert r.status_code == 200
    assert r.json() == {"text": "ab"}


def test_piece_endpoint(client) -> None:
    r = client.post("/v1/piece", json={"id": 88})
    assert r.status_code == 200
    assert r.json() == {"text": "X"}


# --------------------------------------------------------------------------- #
# Inference endpoints
# --------------------------------------------------------------------------- #
def test_next_distribution_returns_ranked_candidates(client) -> None:
    r = client.post("/v1/next_distribution", json={"ids": [97, 98], "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["position"] == 2
    assert data["is_full_vocab"] is True
    assert [c["text"] for c in data["candidates"]] == ["X", "Y"]
    assert data["candidates"][0]["rank"] == 0


def test_score_prompt_returns_per_position_steps(client) -> None:
    """FakeBackend uses the generic Backend.score_prompt which re-evaluates
    per prefix. We just assert the shape (N steps for an N-token prompt,
    last step's chosen is None, watched mapping carries through)."""
    r = client.post(
        "/v1/score_prompt",
        json={"prompt": "ab", "top_k": 3, "watch_ids": [88]},
    )
    assert r.status_code == 200
    steps = r.json()["steps"]
    assert len(steps) == 2  # one per token in "ab"
    assert steps[-1]["chosen"] is None  # trailing "predict next" row
    # watched is encoded as a list of {token_id, candidate}
    last_watched = {w["token_id"]: w["candidate"] for w in steps[-1]["watched"]}
    assert 88 in last_watched


def test_score_prompt_maps_notimplemented_to_400() -> None:
    """A backend whose score_prompt raises NotImplementedError surfaces
    as HTTP 400 rather than a 500 (the canonical chat-only path)."""

    class _ChatOnly(Backend):
        @property
        def capabilities(self):
            from decoding_sandbox.core.types import Capabilities

            return Capabilities(
                name="chat-only",
                full_vocab=False,
                prompt_logprobs=False,
                max_top_logprobs=5,
            )

        def tokenize(self, text):
            return [0]

        def detokenize(self, ids):
            return ""

        def piece(self, tid):
            return ""

        def next_distribution(self, token_ids, top_k, *, watch_ids=()):
            return StepResult(position=len(token_ids), candidates=[], is_full_vocab=False)

        def score_prompt(self, prompt, top_k, watch_ids=None, *, prepend_token_ids=()):
            raise NotImplementedError("chat-only providers cannot score prompts")

    app = make_app(_ChatOnly())
    with TestClient(app) as c:
        r = c.post("/v1/score_prompt", json={"prompt": "hi", "top_k": 5})
    assert r.status_code == 400
    assert "chat-only" in r.json()["detail"]


def test_score_prompt_forwards_prepend_token_ids_to_backend() -> None:
    """The /v1/score_prompt endpoint forwards prepend ids verbatim.

    Without this wiring the frontend's "fill BOS" helper would silently
    no-op on remote dsbx-server-backed backends (dsbx-host-py): the field
    would be parsed off the request, then dropped on the floor before
    reaching the backend. We use a tiny capture-only fake here rather
    than a full FakeBackend so the assertion is exactly "the kwarg
    reached the backend" with no other moving parts.
    """
    captured: dict = {}

    class _Capturing(Backend):
        @property
        def capabilities(self):
            from decoding_sandbox.core.types import Capabilities

            return Capabilities(
                name="capture",
                full_vocab=True,
                prompt_logprobs=True,
                max_top_logprobs=5,
                supports_prepend_token_ids=True,
                bos_token_ids=(42,),
            )

        def tokenize(self, text):
            return [97, 98]

        def detokenize(self, ids):
            return ""

        def piece(self, tid):
            return ""

        def next_distribution(self, token_ids, top_k, *, watch_ids=()):
            return StepResult(position=len(token_ids), candidates=[], is_full_vocab=True)

        def score_prompt(self, prompt, top_k, watch_ids=None, *, prepend_token_ids=()):
            captured["prepend"] = list(prepend_token_ids)
            captured["watch_ids"] = list(watch_ids or [])
            return []

    app = make_app(_Capturing())
    with TestClient(app) as c:
        r = c.post(
            "/v1/score_prompt",
            json={
                "prompt": "x",
                "top_k": 5,
                "watch_ids": [1, 2],
                "prepend_token_ids": [42, 43],
            },
        )
    assert r.status_code == 200
    assert captured["prepend"] == [42, 43]
    assert captured["watch_ids"] == [1, 2]


def test_verify_greedy_endpoint() -> None:
    """A backend with verify_greedy returns ``{accepted, correction}``."""

    class _WithVerify(Backend):
        @property
        def capabilities(self):
            from decoding_sandbox.core.types import Capabilities

            return Capabilities(
                name="vg",
                full_vocab=True,
                prompt_logprobs=True,
                max_top_logprobs=5,
            )

        def tokenize(self, text):
            return [ord(c) for c in text]

        def detokenize(self, ids):
            return "".join(chr(i) for i in ids)

        def piece(self, tid):
            return chr(tid)

        def next_distribution(self, token_ids, top_k, *, watch_ids=()):
            return StepResult(
                position=len(token_ids),
                candidates=[TokenCandidate(88, "X", math.log(0.9), 0)],
                is_full_vocab=True,
            )

        def verify_greedy(self, context_ids, draft_ids):
            # Accept all drafts; emit a bonus token.
            return len(draft_ids), TokenCandidate(89, "Y", math.log(0.5), 0)

    app = make_app(_WithVerify())
    with TestClient(app) as c:
        r = c.post(
            "/v1/verify_greedy",
            json={"context_ids": [97], "draft_ids": [98, 99]},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["accepted"] == 2
    assert data["correction"]["token_id"] == 89


def test_verify_greedy_400_when_unsupported(client) -> None:
    """FakeBackend has no verify_greedy -> 400 with explanatory detail."""
    r = client.post(
        "/v1/verify_greedy",
        json={"context_ids": [97], "draft_ids": [98]},
    )
    assert r.status_code == 400
    assert "verify_greedy" in r.json()["detail"]


# --------------------------------------------------------------------------- #
# SSE /v1/generate/stream
# --------------------------------------------------------------------------- #
def _parse_sse_events(body_text: str) -> list[dict]:
    """Pull every ``data:`` payload out of an SSE response body."""
    events: list[dict] = []
    for chunk in body_text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data:"):
            continue
        payload = chunk[len("data:") :].strip()
        events.append(json.loads(payload))
    return events


def test_generate_stream_emits_step_events_then_done(client) -> None:
    body = {
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 2,
        "top_k": 5,
        "stop_ids": [],
        "seed": 0,
    }
    r = client.post("/v1/generate/stream", json=body)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(r.text)
    # At least one step event + one terminating done event.
    kinds = [e["event"] for e in events]
    assert kinds[-1] == "done"
    step_events = [e for e in events if e["event"] == "step"]
    assert len(step_events) == 2
    # First step's chosen token is the greedy pick from the (97,98) distribution.
    assert step_events[0]["step"]["decision"]["token_text"] == "X"
    # Done event carries the final stop_reason (max_tokens here).
    assert events[-1]["stop_reason"] == "max_tokens"
    assert events[-1].get("error") is None


def test_generate_stream_rejects_custom_sampler(client) -> None:
    body = {
        "prompt": "ab",
        "sampler": {"name": "custom", "params": {}},
        "max_tokens": 1,
        "top_k": 5,
    }
    r = client.post("/v1/generate/stream", json=body)
    assert r.status_code == 400
    assert "custom" in r.json()["detail"]


def test_generate_stream_rejects_unknown_sampler(client) -> None:
    body = {
        "prompt": "ab",
        "sampler": {"name": "nope", "params": {}},
        "max_tokens": 1,
        "top_k": 5,
    }
    r = client.post("/v1/generate/stream", json=body)
    assert r.status_code == 400
    assert "unknown sampler" in r.json()["detail"]


def test_generate_stream_runtime_error_lands_in_done_event() -> None:
    """An exception mid-decode is wrapped as a final ``done`` event with
    an ``error`` field, not a hard HTTP 500 (headers are already
    committed by the time decoding starts)."""

    class _ExplodingBackend(Backend):
        @property
        def capabilities(self):
            from decoding_sandbox.core.types import Capabilities

            return Capabilities(
                name="boom",
                full_vocab=True,
                prompt_logprobs=True,
                max_top_logprobs=5,
            )

        def tokenize(self, text):
            return [ord(c) for c in text]

        def detokenize(self, ids):
            return ""

        def piece(self, tid):
            return chr(tid)

        def next_distribution(self, token_ids, top_k, *, watch_ids=()):
            raise RuntimeError("kaboom")

    app = make_app(_ExplodingBackend())
    body = {
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 2,
        "top_k": 5,
    }
    with TestClient(app) as c:
        r = c.post("/v1/generate/stream", json=body)
    # Stream still returns 200 (headers were committed before the error).
    assert r.status_code == 200
    events = _parse_sse_events(r.text)
    assert events[-1]["event"] == "done"
    assert events[-1]["error"] == "kaboom"


# --------------------------------------------------------------------------- #
# Swappable model slot: /v1/status, /v1/models, /v1/reload
# --------------------------------------------------------------------------- #
def _wait_for_state(c: TestClient, target: str, timeout: float = 5.0) -> dict:
    """Poll /v1/status until it reaches ``target`` (or the timeout fires)."""
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = c.get("/v1/status").json()
        if last["state"] == target:
            return last
        time.sleep(0.02)
    return last


def test_no_preload_starts_empty_and_blocks_inference() -> None:
    """``preload=False`` -> empty slot; inference 409s until a model loads."""
    app = make_app(
        backend_kind="hf",
        builder=lambda _m: _make_fake(),
        preload=False,
    )
    with TestClient(app) as c:
        assert c.get("/v1/status").json()["state"] == "empty"
        info = c.get("/v1/info").json()
        assert info["capabilities"] is None
        assert info["state"] == "empty"
        # Every inference route should refuse with a clean 409.
        assert c.post("/v1/tokenize", json={"text": "ab"}).status_code == 409
        assert c.post("/v1/next_distribution", json={"ids": [97], "top_k": 3}).status_code == 409


def test_reload_lifecycle_loading_then_ready() -> None:
    """A reload moves empty -> loading -> ready and unlocks inference."""
    gate = threading.Event()

    def builder(model):
        # Block until the test lets the build finish so we can observe the
        # intermediate ``loading`` state deterministically.
        gate.wait(timeout=5)
        fake = _make_fake()
        fake.model_path = model or "default.gguf"
        return fake

    app = make_app(backend_kind="llamacpp-py", builder=builder, preload=False)
    with TestClient(app) as c:
        r = c.post("/v1/reload", json={"model": "m1.gguf"})
        assert r.status_code == 200
        assert r.json()["state"] == "loading"
        # Still loading while the gate is closed.
        assert c.get("/v1/status").json()["state"] == "loading"
        # A second reload while one is in progress is rejected.
        assert c.post("/v1/reload", json={"model": "m2.gguf"}).status_code == 409
        # Inference during load -> 409.
        assert c.post("/v1/tokenize", json={"text": "ab"}).status_code == 409
        # Let the build complete.
        gate.set()
        st = _wait_for_state(c, "ready")
        assert st["state"] == "ready"
        assert st["loaded_model"] == "m1.gguf"
        assert st["capabilities"]["name"] == "fake"
        # Inference now works.
        assert c.post("/v1/tokenize", json={"text": "ab"}).status_code == 200


def test_reload_failure_sets_error_state() -> None:
    """A builder that raises leaves the slot in ``error`` with the message."""

    def builder(_model):
        raise RuntimeError("boom while loading")

    app = make_app(backend_kind="hf", builder=builder, preload=False)
    with TestClient(app) as c:
        c.post("/v1/reload", json={"model": "x"})
        st = _wait_for_state(c, "error")
        assert st["state"] == "error"
        assert "boom while loading" in st["error"]
        # Inference reflects the failure (still 409, not 500).
        r = c.post("/v1/tokenize", json={"text": "ab"})
        assert r.status_code == 409
        assert "failed to load" in r.json()["detail"]


def test_models_endpoint_lists_catalogue() -> None:
    entries = [
        S.ServerModelEntry(id="/m/a.gguf", label="a"),
        S.ServerModelEntry(id="/m/b.gguf", label="b"),
    ]
    app = make_app(
        backend_kind="llamacpp-py",
        builder=lambda _m: _make_fake(),
        model_lister=lambda: entries,
        preload=False,
    )
    with TestClient(app) as c:
        data = c.get("/v1/models").json()
    assert data["backend_kind"] == "llamacpp-py"
    assert [m["id"] for m in data["models"]] == ["/m/a.gguf", "/m/b.gguf"]


def test_eager_backend_still_serves_and_reload_swaps() -> None:
    """The legacy adopt path stays ready and can still swap via a builder."""
    first = _make_fake()
    first.model_path = "first.gguf"

    def builder(model):
        nxt = _make_fake()
        nxt.model_path = model or "next.gguf"
        return nxt

    app = make_app(first, backend_kind="llamacpp-py", builder=builder, model="first.gguf")
    with TestClient(app) as c:
        assert c.get("/v1/status").json()["state"] == "ready"
        assert c.get("/v1/info").json()["loaded_model"] == "first.gguf"
        c.post("/v1/reload", json={"model": "second.gguf"})
        st = _wait_for_state(c, "ready")
        assert st["loaded_model"] == "second.gguf"
    # The previous backend was closed during the swap.
    assert first.closed is True


def test_unload_closes_backend_and_marks_slot_empty() -> None:
    backend = _make_fake()
    backend.model_path = "loaded.gguf"
    slot = BackendSlot(backend_kind="llamacpp-py", builder=lambda _m: _make_fake())
    slot.adopt(backend, "loaded.gguf")

    assert slot.status().state == "ready"
    slot.unload()
    status = slot.status()
    assert status.state == "empty"
    assert status.loaded_model is None
    assert backend.closed is True


def test_generate_stream_409_when_no_model() -> None:
    app = make_app(backend_kind="hf", builder=lambda _m: _make_fake(), preload=False)
    body = {
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 1,
        "top_k": 5,
    }
    with TestClient(app) as c:
        r = c.post("/v1/generate/stream", json=body)
    assert r.status_code == 409


# --------------------------------------------------------------------------- #
# Root + miscellany
# --------------------------------------------------------------------------- #
def test_root_lists_endpoints(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "/v1/info" in data["endpoints"]
    assert "/v1/reload" in data["endpoints"]
    assert data["backend_kind"] == "fake-kind"
