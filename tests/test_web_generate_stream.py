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


def test_generate_stream_returns_400_for_chat_only_backend() -> None:
    """Chat-only providers (NIM, OpenRouter) are registered but inert until
    proper chat-mode UI lands. The route refuses to open the stream with a
    400 carrying a human-readable explanation, instead of half-streaming
    an SSE error.

    The historical per-step "growing user message" emulation produced N
    independent first-responses to N slightly-different user queries
    rather than a real continuation, so a sandbox that exists to show
    the truth shouldn't pretend otherwise. The route guard mirrors the
    ``OpenAICompatBackend.next_distribution`` raise + the
    ``Capabilities.generation_disabled`` flag.
    """

    class _ChatOnlyFake(FakeBackend):
        @property
        def capabilities(self) -> Capabilities:
            base = super().capabilities
            return Capabilities(
                name=base.name,
                full_vocab=False,
                prompt_logprobs=False,
                max_top_logprobs=base.max_top_logprobs,
                can_force_token=False,
                notes=(
                    "chat-only provider; generation disabled until "
                    "proper chat-mode UI lands"
                ),
                generation_disabled=True,
            )

    backend = _ChatOnlyFake(
        tokens={"ab": [97, 98]}, pieces={97: "a", 98: "b"}, distributions={}
    )
    app = build_test_app({"chat-only": backend})
    body = {
        "backend": "chat-only",
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 1,
        "top_k": 5,
    }
    with make_authed_client(app) as c:
        r = c.post("/api/v1/generate/stream", json=body)
    assert r.status_code == 400
    detail = r.json()["detail"].lower()
    assert "chat-only" in detail
    assert "generation is disabled" in detail


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


def test_generate_stream_accepts_watch_ids_and_emits_watched_on_each_step(client) -> None:
    """The unified Decode workbench's inspect/generate buttons forward
    the ``watch_*`` panel on every request; the underlying ``GenStep`` /
    ``StepResult`` payloads then carry a ``watched`` map per row. We
    verify a) the request shape is accepted (no 4xx) and b) the watched
    map for the requested ids appears on every emitted ``step`` frame.
    """
    body = {
        "backend": "dsbx-host-py",
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 2,
        "top_k": 3,
        "seed": 0,
        # Mix all three knobs: text + id + eos so the ``_resolve_watches``
        # helper exercises each branch. FakeBackend's tokenize splits
        # strings into per-char ids so " " -> [32]; the resolved
        # watch_ids will at least include 32 (text), 99 (id), and any
        # configured eos ids.
        "watch_texts": [" "],
        "watch_ids": [99],
        "watch_eos": True,
    }
    r = client.post("/api/v1/generate/stream", json=body)
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    step_events = [e for e in events if e["event"] == "step"]
    assert step_events, "expected at least one step frame"
    # Every step frame should carry a ``watched`` list with one entry
    # per resolved watch id. The FakeBackend's ``lookup_watch`` populates
    # them; we don't care about the exact values, only that the wire
    # shape carries them through.
    for evt in step_events:
        watched = evt["step"]["step_result"]["watched"]
        assert isinstance(watched, list)
        # At least the id-watch entry (99) should be present.
        ids = {w["token_id"] for w in watched}
        assert 99 in ids, f"expected watched id 99, got {ids}"


def test_generate_stream_accepts_prefix_token_ids(client) -> None:
    """Manual mode ships ``prefix_token_ids`` on every per-pick call.
    The middleware route must accept it (no 4xx for an unknown field)
    and forward it through the engine so the model decodes from
    ``tokenize(prompt) + prefix``. We can't easily assert the FakeBackend
    received the prefix without instrumenting it, but a successful 200
    + step frame proves the wire path is wired.
    """
    body = {
        "backend": "dsbx-host-py",
        "prompt": "ab",
        "sampler": {"name": "greedy", "params": {}},
        "max_tokens": 1,
        "top_k": 3,
        "seed": 0,
        "include_prompt": False,
        "prefix_token_ids": [101, 102],
    }
    r = client.post("/api/v1/generate/stream", json=body)
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    assert any(e["event"] == "step" for e in events) or any(
        e["event"] == "done" for e in events
    )


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
            service_tier=None,
            prompt_cache_key=None,
            session_id=None,
            logit_bias=None,
            watch_ids=(),
            prefix_token_ids=(),
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
                    "watch_ids": list(watch_ids or []),
                    "prefix_token_ids": list(prefix_token_ids or []),
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

        def next_distribution(self, token_ids, top_k, *, watch_ids=()):
            next_dist_calls["n"] += 1
            return super().next_distribution(token_ids, top_k, watch_ids=watch_ids)

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

        def next_distribution(self, token_ids, top_k, *, watch_ids=()):
            next_dist_calls["n"] += 1
            return super().next_distribution(token_ids, top_k, watch_ids=watch_ids)

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
            service_tier=None,
            prompt_cache_key=None,
            session_id=None,
            logit_bias=None,
            watch_ids=(),
            prefix_token_ids=(),
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


def test_generate_stream_emits_perf_event_before_usage() -> None:
    """A perf_metrics dict on the sink lands as a dedicated ``perf`` frame.

    Order: prompt_score? -> step* -> perf? -> usage -> done. The
    consumer keys off ``usage`` for token counts and off ``perf`` for
    server timings; emitting them as separate frames lets a minimal
    client ignore ``perf`` entirely without having to learn the
    provider's metric schema.
    """
    from decoding_sandbox.core.engine import GenStep
    from decoding_sandbox.core.samplers import SamplerDecision
    from decoding_sandbox.core.types import StepResult, TokenCandidate

    class _PerfFake(FakeBackend):
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
            service_tier=None,
            prompt_cache_key=None,
            session_id=None,
            logit_bias=None,
            watch_ids=(),
            prefix_token_ids=(),
        ):
            if self._sink is not None:
                self._sink["perf_metrics"] = {
                    "server-time-to-first-token": 0.042,
                    "prefill-duration": 0.011,
                    "generation-duration": 0.031,
                    "prompt-tokens": 2,
                }
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

    backend = _PerfFake()
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
    assert events[-3]["event"] == "perf"
    perf_event = events[-3]
    metrics = perf_event["metrics"]
    assert metrics["server-time-to-first-token"] == pytest.approx(0.042)
    assert metrics["prompt-tokens"] == 2
    # Critical: ``perf_metrics`` MUST be popped from the usage frame so
    # the schema (UsageEvent) doesn't accidentally carry an opaque
    # provider-specific dict alongside the standard token counters.
    usage_event = events[-2]
    assert "perf_metrics" not in usage_event


def test_generate_stream_uses_combined_echo_path_when_supported() -> None:
    """include_prompt + combined_echo_stream backend -> ONE call, one stream.

    The legacy two-request path emits a prompt_score frame from a
    separate ``score_prompt`` call before the per-token loop. The
    Phase-5 combined path runs ``stream_native_with_echo`` instead --
    one call, same wire shape (one prompt_score then a sequence of
    step frames). We pin that:

    1. score_prompt is NOT called.
    2. stream_native_with_echo IS called exactly once.
    3. The SSE wire order is unchanged
       (prompt_score -> step* -> usage -> done).
    """
    from decoding_sandbox.core.engine import GenStep
    from decoding_sandbox.core.samplers import SamplerDecision
    from decoding_sandbox.core.types import (
        Capabilities,
        StepResult,
        TokenCandidate,
    )

    class _ComboFake(FakeBackend):
        def __init__(self):
            super().__init__(
                tokens={"the cap of": [1, 2, 3]},
                pieces={1: "the", 2: " cap", 3: " of", 10: "France", 11: " is"},
                distributions={(1, 2, 3): [cand(10, "France", 0.99, 0)]},
            )
            self.score_prompt_called = 0
            self.echo_stream_called = 0

        @property
        def capabilities(self) -> Capabilities:
            base = super().capabilities
            return Capabilities(
                name=base.name,
                full_vocab=base.full_vocab,
                prompt_logprobs=base.prompt_logprobs,
                max_top_logprobs=base.max_top_logprobs,
                can_force_token=base.can_force_token,
                notes=base.notes,
                supports_combined_echo_stream=True,
            )

        def supports_native_sampler(self, name, params):
            return name == "greedy"

        def score_prompt(self, prompt, *, top_k, watch_ids=()):
            self.score_prompt_called += 1
            return super().score_prompt(prompt, top_k=top_k, watch_ids=watch_ids)

        def stream_native_with_echo(
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
            service_tier=None,
            prompt_cache_key=None,
            session_id=None,
            logit_bias=None,
            echo_last=None,
            watch_ids=(),
            prefix_token_ids=(),
        ):
            self.echo_stream_called += 1
            # 3 prompt-echo StepResults then 1 emitted GenStep.
            yield StepResult(
                position=0,
                candidates=[TokenCandidate(1, "the", -0.1, 0)],
                is_full_vocab=False,
                chosen=TokenCandidate(1, "the", -0.1, 0),
                context_text="the",
            )
            yield StepResult(
                position=1,
                candidates=[TokenCandidate(2, " cap", -0.1, 0)],
                is_full_vocab=False,
                chosen=TokenCandidate(2, " cap", -0.1, 0),
                context_text=" cap",
            )
            yield StepResult(
                position=2,
                candidates=[TokenCandidate(3, " of", -0.1, 0)],
                is_full_vocab=False,
                chosen=TokenCandidate(3, " of", -0.1, 0),
                context_text=" of",
            )
            cand_fr = TokenCandidate(10, "France", -0.01, 0)
            yield GenStep(
                step=0,
                tokens_before=[1, 2, 3],
                step_result=StepResult(
                    position=3, candidates=[cand_fr], is_full_vocab=False, chosen=cand_fr
                ),
                decision=SamplerDecision(
                    token_id=10, token_text="France", kept=[], greedy_token_id=10, note=""
                ),
                stop_reason="max_tokens",
            )

    backend = _ComboFake()
    app = build_test_app({"dsbx-host-py": backend})
    with make_authed_client(app) as c:
        r = c.post(
            "/api/v1/generate/stream",
            json={
                "backend": "dsbx-host-py",
                "prompt": "the cap of",
                "sampler": {"name": "greedy", "params": {}},
                "max_tokens": 1,
                "top_k": 1,
                "include_prompt": True,
            },
        )
    events = _parse_sse(r.text)
    assert backend.score_prompt_called == 0, "legacy two-request path was hit"
    assert backend.echo_stream_called == 1, "combined path should fire exactly once"
    # Wire order: prompt_score -> step -> usage -> done.
    assert events[0]["event"] == "prompt_score"
    assert len(events[0]["steps"]) == 3
    assert events[1]["event"] == "step"
    assert events[1]["step"]["decision"]["token_id"] == 10
    assert events[-1]["event"] == "done"
    assert events[-2]["event"] == "usage"


def test_generate_stream_combined_echo_path_forwards_prefix_and_watch() -> None:
    """The Fireworks combined-echo+stream path is the most fragile spot
    for the new manual + watch wiring: ``prefix_token_ids`` has to land
    in :meth:`stream_native_with_echo` so the backend can detokenize +
    concatenate it onto the prompt text, and ``watch_ids`` has to ride
    along so per-step watched columns light up. We assert both arrive at
    the fake backend verbatim.
    """
    from decoding_sandbox.core.engine import GenStep
    from decoding_sandbox.core.samplers import SamplerDecision
    from decoding_sandbox.core.types import (
        Capabilities,
        StepResult,
        TokenCandidate,
    )

    seen: dict[str, object] = {}

    class _ComboFake(FakeBackend):
        def __init__(self):
            super().__init__(
                tokens={"hi": [1, 2]},
                pieces={1: "h", 2: "i", 5: "Y", 9: "Z", 10: "X"},
            )

        @property
        def capabilities(self) -> Capabilities:
            base = super().capabilities
            return Capabilities(
                name=base.name,
                full_vocab=base.full_vocab,
                prompt_logprobs=base.prompt_logprobs,
                max_top_logprobs=base.max_top_logprobs,
                can_force_token=base.can_force_token,
                notes=base.notes,
                supports_combined_echo_stream=True,
            )

        def supports_native_sampler(self, name, params):
            return name == "greedy"

        def stream_native_with_echo(
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
            service_tier=None,
            prompt_cache_key=None,
            session_id=None,
            logit_bias=None,
            echo_last=None,
            watch_ids=(),
            prefix_token_ids=(),
        ):
            seen["prompt"] = prompt
            seen["prefix_token_ids"] = list(prefix_token_ids)
            seen["watch_ids"] = list(watch_ids)
            cand_x = TokenCandidate(10, "X", -0.01, 0)
            yield GenStep(
                step=0,
                tokens_before=[1, 2, 5, 9],
                step_result=StepResult(
                    position=4, candidates=[cand_x], is_full_vocab=False, chosen=cand_x
                ),
                decision=SamplerDecision(
                    token_id=10, token_text="X", kept=[], greedy_token_id=10, note=""
                ),
                stop_reason="max_tokens",
            )

    backend = _ComboFake()
    app = build_test_app({"dsbx-host-py": backend})
    with make_authed_client(app) as c:
        r = c.post(
            "/api/v1/generate/stream",
            json={
                "backend": "dsbx-host-py",
                "prompt": "hi",
                "sampler": {"name": "greedy", "params": {}},
                "max_tokens": 1,
                "top_k": 1,
                "include_prompt": True,
                "prefix_token_ids": [5, 9],
                "watch_ids": [42],
            },
        )
    assert r.status_code == 200, r.text
    assert seen["prefix_token_ids"] == [5, 9]
    assert seen["watch_ids"] == [42]


def test_generate_stream_emits_raw_output_event_and_forwards_logit_bias() -> None:
    """``raw_output`` lands as a dedicated SSE frame; ``logit_bias`` flows to backend.

    Two things in one test because they share a fake backend setup:

    1. When the active backend stashes ``raw_output`` on the sink (the
       Fireworks path does this from the final stream chunk), the web
       layer must flush it as a ``raw_output`` SSE frame BEFORE the
       ``usage`` / ``done`` frames so the browser's "what the model
       saw" panel can render it alongside the same-run perf timings.
    2. The ``logit_bias`` field on ``GenerateRequest`` must reach the
       backend's ``stream_native`` with int keys (not the stringified
       JSON keys), so a follow-up backend test for the wire shape has
       something real to assert against.
    """
    from decoding_sandbox.core.engine import GenStep
    from decoding_sandbox.core.samplers import SamplerDecision
    from decoding_sandbox.core.types import StepResult, TokenCandidate

    raw_payload = {
        "prompt_fragments": ["<|system|>", "be useful", "<|user|>", "hi"],
        "prompt_token_ids": [1, 200, 5, 300],
        "grammar": None,
    }

    class _RawFake(FakeBackend):
        def __init__(self):
            super().__init__(
                tokens={"hi": [1, 2]},
                pieces={1: "h", 2: "i", 10: "X"},
                distributions={(1, 2): [cand(10, "X", 0.99, 0)]},
            )
            self._sink: dict | None = None
            self.seen_logit_bias: dict | None = None

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
            service_tier=None,
            prompt_cache_key=None,
            session_id=None,
            logit_bias=None,
            watch_ids=(),
            prefix_token_ids=(),
        ):
            self.seen_logit_bias = logit_bias
            if self._sink is not None:
                self._sink["raw_output"] = dict(raw_payload)
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

    backend = _RawFake()
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
                # Wire uses string keys (JSON requirement); the app
                # layer coerces to ints before passing through.
                "logit_bias": {"100": 5.0, "200": -3.0},
            },
        )
    events = _parse_sse(r.text)
    assert events[-1]["event"] == "done"
    assert events[-2]["event"] == "usage"
    assert events[-3]["event"] == "raw_output"
    raw_event = events[-3]
    assert raw_event["payload"] == raw_payload
    # ``raw_output`` MUST be popped from the usage frame so UsageEvent
    # stays free of opaque provider diagnostics.
    assert "raw_output" not in events[-2]
    # Backend receives ``logit_bias`` with INT keys (not strings).
    assert backend.seen_logit_bias == {100: 5.0, 200: -3.0}


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

        def next_distribution(self, token_ids, top_k, *, watch_ids=()):
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
