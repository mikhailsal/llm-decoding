"""Tests for /api/v1/generate/stream (SSE).

We parse the ``text/event-stream`` body line by line and assert:

- Each step is one ``{"event":"step", "step": WireGenStep}`` frame.
- The terminator is exactly one ``{"event":"done", "stop_reason": ...}`` frame.
- Custom samplers are rejected at the 400 boundary, never streamed.
- Unknown samplers are rejected at the 400 boundary too.
- Mid-stream backend exceptions surface as ``done.error`` without crashing
  the connection (200 status, structured error in the final frame).
"""

from __future__ import annotations

import json

import pytest

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.types import Capabilities
from tests.fakes import FakeBackend, cand
from tests.web_helpers import build_test_app, make_authed_client


def _parse_sse(body: str) -> list[dict]:
    """Pull every ``data:`` payload out of an SSE response body."""
    events: list[dict] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data:"):
            continue
        payload = chunk[len("data:") :].strip()
        events.append(json.loads(payload))
    return events


def _backend() -> FakeBackend:
    return FakeBackend(
        tokens={"ab": [97, 98]},
        pieces={97: "a", 98: "b", 88: "X", 89: "Y"},
        distributions={
            (97, 98): [cand(88, "X", 0.6, 0), cand(89, "Y", 0.4, 1)],
            (97, 98, 88): [cand(88, "X", 0.55, 0), cand(89, "Y", 0.45, 1)],
            (97, 98, 88, 88): [cand(88, "X", 0.55, 0), cand(89, "Y", 0.45, 1)],
        },
        eos_token_ids=(99,),
    )


@pytest.fixture
def client():
    app = build_test_app({"dsbx-host-py": _backend()})
    with make_authed_client(app) as c:
        yield c


def test_generate_stream_emits_step_events_and_done(client) -> None:
    body = {
        "backend": "dsbx-host-py",
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 2,
        "top_k": 5,
        "stop_ids": [],
        "seed": 0,
    }
    r = client.post("/api/v1/generate/stream", json=body)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    step_events = [e for e in events if e["event"] == "step"]
    assert len(step_events) == 2
    # Greedy picks the rank-0 candidate "X" at each step.
    assert step_events[0]["step"]["decision"]["token_text"] == "X"
    # The terminator is exactly one ``done`` event.
    assert events[-1]["event"] == "done"
    assert events[-1]["stop_reason"] == "max_tokens"
    assert events[-1].get("error") is None


def test_generate_stream_stops_on_user_stop_token(client) -> None:
    """``stop_ids`` halts generation as soon as that id is chosen."""
    body = {
        "backend": "dsbx-host-py",
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 5,
        "top_k": 5,
        "stop_ids": [88],  # X is greedy at every step -> first emission stops
        "seed": 0,
    }
    r = client.post("/api/v1/generate/stream", json=body)
    events = _parse_sse(r.text)
    step_events = [e for e in events if e["event"] == "step"]
    assert len(step_events) == 1
    assert events[-1]["event"] == "done"
    assert events[-1]["stop_reason"] == "user_stop"


def test_generate_stream_resolves_stop_texts_to_ids(client) -> None:
    """``stop_texts`` is tokenized server-side; a single-token match stops."""
    backend = _backend()
    # Add the stop string to the fake tokenizer so it tokenizes to one id.
    backend.tokens["STOP"] = [88]
    app = build_test_app({"dsbx-host-py": backend})
    body = {
        "backend": "dsbx-host-py",
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 5,
        "top_k": 5,
        "stop_texts": ["STOP"],
        "seed": 0,
    }
    with make_authed_client(app) as c:
        r = c.post("/api/v1/generate/stream", json=body)
    events = _parse_sse(r.text)
    assert events[-1]["stop_reason"] == "user_stop"


def test_generate_stream_rejects_custom_sampler(client) -> None:
    body = {
        "backend": "dsbx-host-py",
        "prompt": "ab",
        "sampler": {"name": "custom", "params": {}},
        "max_tokens": 1,
        "top_k": 5,
    }
    r = client.post("/api/v1/generate/stream", json=body)
    assert r.status_code == 400
    assert "custom" in r.json()["detail"].lower()


def test_generate_stream_rejects_unknown_sampler(client) -> None:
    body = {
        "backend": "dsbx-host-py",
        "prompt": "ab",
        "sampler": {"name": "no-such-sampler", "params": {}},
        "max_tokens": 1,
        "top_k": 5,
    }
    r = client.post("/api/v1/generate/stream", json=body)
    assert r.status_code == 400
    assert "unknown sampler" in r.json()["detail"]


def test_generate_stream_top_p_works_with_params(client) -> None:
    """Sampler params reach the builder and produce a valid stream."""
    body = {
        "backend": "dsbx-host-py",
        "prompt": "ab",
        "sampler": {"name": "top_p", "params": {"top_p": 0.9, "temperature": 1.0}},
        "max_tokens": 1,
        "top_k": 10,
        "seed": 42,
    }
    r = client.post("/api/v1/generate/stream", json=body)
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert any(e["event"] == "step" for e in events)
    assert events[-1]["event"] == "done"


def test_generate_stream_include_prompt_emits_prompt_score_first(client) -> None:
    """``include_prompt=true`` produces a single ``prompt_score`` frame before
    the regular ``step`` frames, with one entry per prompt token. The
    browser uses this to show prompt-token logits in the same table as
    generation, which was previously inspect-only."""
    body = {
        "backend": "dsbx-host-py",
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 1,
        "top_k": 5,
        "seed": 0,
        "include_prompt": True,
    }
    r = client.post("/api/v1/generate/stream", json=body)
    assert r.status_code == 200
    events = _parse_sse(r.text)
    # First non-empty frame is prompt_score; then exactly one step and a done.
    assert events[0]["event"] == "prompt_score"
    assert isinstance(events[0]["steps"], list)
    # FakeBackend.score_prompt returns one StepResult per prompt token.
    assert len(events[0]["steps"]) == len("ab")
    assert events[0]["prompt_logprobs"] is True
    step_events = [e for e in events if e["event"] == "step"]
    assert len(step_events) == 1
    assert events[-1]["event"] == "done"


def test_generate_stream_default_does_not_emit_prompt_score(client) -> None:
    """Without ``include_prompt`` the wire shape is identical to before."""
    body = {
        "backend": "dsbx-host-py",
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 1,
        "top_k": 5,
        "seed": 0,
    }
    r = client.post("/api/v1/generate/stream", json=body)
    events = _parse_sse(r.text)
    assert all(e["event"] != "prompt_score" for e in events)


# --------------------------------------------------------------------------- #
# Native-streaming opt-in (backend implements supports_native_sampler + stream_native)
# --------------------------------------------------------------------------- #
def test_generate_stream_uses_native_path_when_backend_opts_in() -> None:
    """A backend advertising ``supports_native_sampler`` skips the per-step loop.

    This is the wire-level check on the Fireworks 429 fix: when the
    backend signals it can run the sampler server-side, the middleware
    issues a SINGLE call to ``stream_native`` and forwards its
    ``GenStep`` events as SSE frames -- instead of looping
    ``next_distribution`` once per generated token. If this regresses,
    we're back to spamming providers with N requests per generate.
    """
    from decoding_sandbox.core.engine import GenStep
    from decoding_sandbox.core.samplers import SamplerDecision
    from decoding_sandbox.core.types import StepResult, TokenCandidate

    next_dist_calls = {"n": 0}

    class _NativeFake(FakeBackend):
        def __init__(self):
            super().__init__(
                tokens={"hi": [1, 2]},
                pieces={1: "h", 2: "i", 10: "X"},
                distributions={(1, 2): [cand(10, "X", 0.99, 0)]},
            )
            self.native_calls: list[dict] = []

        def supports_native_sampler(self, name, params):  # noqa: D401
            return name in {"greedy", "top_p"}

        def stream_native(
            self,
            prompt,
            *,
            sampler_name,
            sampler_params,
            max_tokens,
            top_k,
            stop_ids=None,
            seed=0,
            respect_eos=True,
        ):
            self.native_calls.append(
                {
                    "prompt": prompt,
                    "sampler_name": sampler_name,
                    "sampler_params": sampler_params,
                    "max_tokens": max_tokens,
                    "top_k": top_k,
                    "stop_ids": list(stop_ids or []),
                    "seed": seed,
                    "respect_eos": respect_eos,
                }
            )
            cand_x = TokenCandidate(token_id=10, text="X", logprob=-0.01, rank=0)
            yield GenStep(
                step=0,
                tokens_before=[1, 2],
                step_result=StepResult(
                    position=2,
                    candidates=[cand_x],
                    is_full_vocab=False,
                    chosen=cand_x,
                ),
                decision=SamplerDecision(
                    token_id=10,
                    token_text="X",
                    kept=[],
                    greedy_token_id=10,
                    note="greedy (server-side)",
                ),
                stop_reason="max_tokens",
            )

        def next_distribution(self, token_ids, top_k):
            next_dist_calls["n"] += 1
            return super().next_distribution(token_ids, top_k)

    backend = _NativeFake()
    app = build_test_app({"dsbx-host-py": backend})
    with make_authed_client(app) as c:
        r = c.post(
            "/api/v1/generate/stream",
            json={
                "backend": "dsbx-host-py",
                "prompt": "hi",
                "sampler": {"name": "greedy", "params": {}},
                "max_tokens": 5,
                "top_k": 5,
            },
        )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    step_events = [e for e in events if e["event"] == "step"]
    assert len(step_events) == 1
    assert step_events[0]["step"]["decision"]["token_text"] == "X"
    assert "server-side" in step_events[0]["step"]["decision"]["note"]
    assert events[-1]["event"] == "done"
    assert events[-1]["stop_reason"] == "max_tokens"
    # The native path was taken: exactly one call, zero next_distribution hits.
    assert len(backend.native_calls) == 1
    assert backend.native_calls[0]["sampler_name"] == "greedy"
    assert next_dist_calls["n"] == 0


def test_generate_stream_falls_back_to_per_step_when_native_says_no() -> None:
    """When ``supports_native_sampler`` returns False the loop path runs.

    Critically, custom samplers must NEVER end up in the native branch
    (we'd silently change behaviour); this test pins that the fallback
    still goes through ``next_distribution`` even if the backend
    *declares* the capability for some samplers.
    """
    next_dist_calls = {"n": 0}
    native_calls = {"n": 0}

    class _PickyFake(FakeBackend):
        def __init__(self):
            super().__init__(
                tokens={"ab": [97, 98]},
                pieces={97: "a", 98: "b", 88: "X"},
                distributions={
                    (97, 98): [cand(88, "X", 0.9, 0)],
                    (97, 98, 88): [cand(88, "X", 0.9, 0)],
                },
            )

        def supports_native_sampler(self, name, params):
            return False  # decline everything; loop must run

        def stream_native(self, *args, **kwargs):
            native_calls["n"] += 1
            return iter([])

        def next_distribution(self, token_ids, top_k):
            next_dist_calls["n"] += 1
            return super().next_distribution(token_ids, top_k)

    backend = _PickyFake()
    app = build_test_app({"dsbx-host-py": backend})
    with make_authed_client(app) as c:
        r = c.post(
            "/api/v1/generate/stream",
            json={
                "backend": "dsbx-host-py",
                "prompt": "ab",
                "sampler": {"name": "greedy", "params": {}},
                "max_tokens": 2,
                "top_k": 5,
            },
        )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert len([e for e in events if e["event"] == "step"]) == 2
    assert native_calls["n"] == 0
    assert next_dist_calls["n"] == 2


def test_generate_stream_emits_usage_event_before_done(client) -> None:
    """Every generate stream now ends with ``usage`` then ``done``.

    For the local (HF/llamacpp_py/dsbx-host-py) path the backend doesn't
    implement :class:`UsageAware`, so ``requests`` stays 0 and the
    stream layer fills ``prompt_tokens`` from ``backend.tokenize`` and
    ``completion_tokens`` from the number of emitted steps. We pin
    the wire shape here so the UI's counter renderer can rely on it.
    """
    body = {
        "backend": "dsbx-host-py",
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 2,
        "top_k": 5,
        "stop_ids": [],
        "seed": 0,
    }
    r = client.post("/api/v1/generate/stream", json=body)
    events = _parse_sse(r.text)
    # The terminator is still exactly one ``done`` -- the new ``usage``
    # frame goes immediately before it.
    assert events[-1]["event"] == "done"
    assert events[-2]["event"] == "usage"
    u = events[-2]
    # Non-UsageAware backend: zero HTTP requests, tokens computed locally.
    assert u["requests"] == 0
    assert u["prompt_tokens"] == 2  # "ab" -> [97, 98]
    assert u["completion_tokens"] == 2  # max_tokens=2, both emitted
    assert u["total_tokens"] == 4
    assert isinstance(u.get("notes"), list)


def test_generate_stream_usage_event_records_native_backend_requests() -> None:
    """A :class:`UsageAware` backend writes ``requests`` into the sink.

    Mirrors what ``OpenAICompatBackend`` does in production: as each
    HTTP attempt fires (success or 429-retry), the backend bumps the
    counter on the bound sink. After the stream completes the web
    layer emits the populated sink as a ``usage`` event so the user
    sees real provider RPS pressure instead of "0 requests" for what
    might be a 20-token storm of per-step calls.
    """
    from decoding_sandbox.core.engine import GenStep
    from decoding_sandbox.core.samplers import SamplerDecision
    from decoding_sandbox.core.types import StepResult, TokenCandidate

    class _AwareFake(FakeBackend):
        def __init__(self):
            super().__init__(
                tokens={"hi": [1, 2]},
                pieces={1: "h", 2: "i", 10: "X"},
                distributions={(1, 2): [cand(10, "X", 0.99, 0)]},
            )
            self._sink = None

        def set_active_usage(self, sink):
            self._sink = sink

        def supports_native_sampler(self, name, params):
            return name == "greedy"

        def stream_native(
            self,
            prompt,
            *,
            sampler_name,
            sampler_params,
            max_tokens,
            top_k,
            stop_ids=None,
            seed=0,
            respect_eos=True,
        ):
            # Simulate two HTTP attempts (e.g. 429 -> 200) and a server-
            # reported usage block. Both must land in the bound sink.
            if self._sink is not None:
                self._sink["requests"] = self._sink.get("requests", 0) + 2
                self._sink["prompt_tokens"] = (self._sink.get("prompt_tokens") or 0) + 7
                self._sink["completion_tokens"] = (self._sink.get("completion_tokens") or 0) + 1
            cand_x = TokenCandidate(token_id=10, text="X", logprob=-0.01, rank=0)
            yield GenStep(
                step=0,
                tokens_before=[1, 2],
                step_result=StepResult(
                    position=2, candidates=[cand_x], is_full_vocab=False, chosen=cand_x
                ),
                decision=SamplerDecision(
                    token_id=10, token_text="X", kept=[], greedy_token_id=10, note=""
                ),
                stop_reason="max_tokens",
            )

    backend = _AwareFake()
    app = build_test_app({"dsbx-host-py": backend})
    with make_authed_client(app) as c:
        r = c.post(
            "/api/v1/generate/stream",
            json={
                "backend": "dsbx-host-py",
                "prompt": "hi",
                "sampler": {"name": "greedy", "params": {}},
                "max_tokens": 1,
                "top_k": 5,
            },
        )
    events = _parse_sse(r.text)
    assert events[-1]["event"] == "done"
    assert events[-2]["event"] == "usage"
    u = events[-2]
    # Backend's writes win over local fallbacks where present.
    assert u["requests"] == 2
    assert u["prompt_tokens"] == 7
    assert u["completion_tokens"] == 1
    # ``total_tokens`` is computed by the streamer when the backend
    # didn't supply it explicitly.
    assert u["total_tokens"] == 8
    # The sink must be unbound after the call so the next stream
    # doesn't accrete onto our dict.
    assert backend._sink is None


def test_generate_stream_runtime_error_lands_in_done() -> None:
    """An exception in the engine mid-decode is wrapped as a done.error
    frame rather than a hard 500 -- streaming response has already
    committed headers by then."""

    class _Exploding(Backend):
        @property
        def capabilities(self) -> Capabilities:
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

        def next_distribution(self, token_ids, top_k):
            raise RuntimeError("kaboom")

    app = build_test_app({"boom": _Exploding()})
    with make_authed_client(app) as c:
        r = c.post(
            "/api/v1/generate/stream",
            json={
                "backend": "boom",
                "prompt": "ab",
                "sampler": {"name": "greedy", "params": {}},
                "max_tokens": 2,
                "top_k": 5,
            },
        )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert events[-1]["event"] == "done"
    assert events[-1]["error"] == "kaboom"
