"""Tests for the HTTP-talking backends (OpenAI-compat and llama.cpp).

We patch in a ``MockHTTPClient`` (see ``tests/fakes.py``) so we exercise the
real JSON-parsing logic without standing up servers. Every test asserts both
(a) the request shape we send and (b) the StepResult we build from the canned
response.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager

import pytest

from decoding_sandbox.backends import llamacpp as llamacpp_mod
from decoding_sandbox.backends import openai_compat as oc_mod
from decoding_sandbox.backends.llamacpp import LlamaCppBackend
from decoding_sandbox.backends.openai_compat import OpenAICompatBackend
from decoding_sandbox.core.config import ProviderConfig
from tests.fakes import MockHTTPClient, MockResponse


# --------------------------------------------------------------------------- #
# OpenAICompatBackend
# --------------------------------------------------------------------------- #
def _make_oc_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    routes: dict[tuple[str, str], object],
    has_completions: bool = True,
    supports_prompt_logprobs: bool = False,
    require_parameters: bool = False,
    max_top: int = 5,
    max_retries: int = 0,
    sleeps: list[float] | None = None,
) -> tuple[OpenAICompatBackend, MockHTTPClient]:
    """Build an OpenAICompatBackend wired to a MockHTTPClient.

    ``max_retries`` defaults to 0 so existing tests stay deterministic
    (one shot, raises on non-2xx). Tests that exercise retry pass it
    explicitly along with a ``sleeps`` list that captures wait values
    instead of actually sleeping.
    """
    mock = MockHTTPClient(routes)
    monkeypatch.setattr(oc_mod.httpx, "Client", lambda **kw: mock)
    prov = ProviderConfig(
        name="test",
        base_url="https://api.test/v1",
        api_key_env="TEST_API_KEY",
        default_model="test/m",
        max_top_logprobs=max_top,
        supports_prompt_logprobs=supports_prompt_logprobs,
        require_parameters=require_parameters,
        has_completions=has_completions,
    )
    sleep_fn = sleeps.append if sleeps is not None else (lambda _w: None)
    backend = OpenAICompatBackend(
        prov, model="test/m", max_retries=max_retries, sleep=sleep_fn
    )
    return backend, mock


def test_openai_compat_next_distribution_completions_parses_dict(monkeypatch) -> None:
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/completions"): {
                "choices": [
                    {
                        "logprobs": {
                            "top_logprobs": [{" Paris": -0.5, " London": -3.0, " Berlin": -4.0}]
                        }
                    }
                ]
            }
        },
        has_completions=True,
    )

    prompt_ids = backend.tokenize("The capital of France is")
    step = backend.next_distribution(prompt_ids, top_k=3)

    assert len(step.candidates) == 3
    assert step.candidates[0].text == " Paris"
    assert step.candidates[0].logprob == pytest.approx(-0.5)
    assert step.candidates[0].rank == 0
    # All texts must be interned to stable ids and the top-1 ranks above others.
    assert step.candidates[1].logprob > step.candidates[2].logprob
    # Request payload sanity:
    assert mock.calls[-1]["url"] == "/completions"
    assert mock.calls[-1]["json"]["model"] == "test/m"
    assert mock.calls[-1]["json"]["logprobs"] == 3


def test_openai_compat_next_distribution_chat_parses_list(monkeypatch) -> None:
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/chat/completions"): {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "top_logprobs": [
                                        {"token": "yes", "logprob": -0.2},
                                        {"token": "no", "logprob": -2.0},
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        },
        has_completions=False,
    )

    step = backend.next_distribution(backend.tokenize("Q"), top_k=2)

    assert [c.text for c in step.candidates] == ["yes", "no"]
    assert step.candidates[0].rank == 0
    assert mock.calls[-1]["url"] == "/chat/completions"


def test_openai_compat_clamps_top_k_to_provider_cap(monkeypatch) -> None:
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={("POST", "/completions"): {"choices": [{"logprobs": {"top_logprobs": [{}]}}]}},
        has_completions=True,
        max_top=3,
    )

    backend.next_distribution(backend.tokenize("x"), top_k=100)

    assert mock.calls[-1]["json"]["logprobs"] == 3


def test_openai_compat_injects_require_parameters(monkeypatch) -> None:
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/chat/completions"): {
                "choices": [{"logprobs": {"content": [{"top_logprobs": []}]}}]
            }
        },
        has_completions=False,
        require_parameters=True,
    )

    backend.next_distribution(backend.tokenize("x"), top_k=2)

    body = mock.calls[-1]["json"]
    assert body["provider"]["require_parameters"] is True


def test_openai_compat_score_prompt_uses_echo_and_records_actual(monkeypatch) -> None:
    """Per the Backend protocol: prompt rows carry actuals; the trailing
    row is the model's ``(predict next)`` slot with ``chosen=None``.

    The mock responds as if echo+max_tokens=1 returned 4 tokens for a
    3-token prompt: ``["The", " cap", "ital", " city"]`` -- the last
    one is the model's first-generation prediction. After the fix the
    backend returns 3 ``StepResult``s: positions 1 and 2 carry the
    actual prompt tokens (" cap", "ital"), and position 3 has
    ``chosen=None`` because there is no "actual" prompt token at that
    slot. Its ``candidates`` still carry the model's top-K at the
    post-prompt position, which the inspect UI renders as ``(predict
    next)`` and the generate path no longer double-counts in its
    running-completion view.
    """
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/completions"): {
                "choices": [
                    {
                        "logprobs": {
                            "tokens": ["The", " cap", "ital", " city"],
                            "token_logprobs": [None, -1.0, -2.0, -0.5],
                            "top_logprobs": [
                                None,
                                {" cap": -1.0, " other": -3.0},
                                {"ital": -2.0, "stuff": -4.0},
                                {" city": -0.5, " is": -1.7, " of": -2.1},
                            ],
                        }
                    }
                ]
            }
        },
        has_completions=True,
        supports_prompt_logprobs=True,
    )

    steps = backend.score_prompt("The capital", top_k=3, watch_ids=[])

    assert len(steps) == 3
    s1, s2, s3 = steps
    # First two: real prompt positions, ``chosen`` = actual prompt token.
    assert s1.context_text == "The"
    assert s1.chosen is not None
    assert s1.chosen.text == " cap"
    assert s1.chosen.logprob == pytest.approx(-1.0)
    assert s2.chosen is not None
    assert s2.chosen.text == "ital"
    # Trailing prediction row: chosen=None, candidates still populated.
    assert s3.chosen is None
    assert s3.position == 3
    assert s3.context_text == "ital"
    assert len(s3.candidates) == 3
    assert s3.candidates[0].text == " city"
    body = mock.calls[-1]["json"]
    assert body["echo"] is True
    assert body["prompt"] == "The capital"


def test_openai_compat_score_prompt_raises_on_chat_only_provider(monkeypatch) -> None:
    backend, _ = _make_oc_backend(
        monkeypatch,
        routes={},
        has_completions=False,
        supports_prompt_logprobs=False,
    )

    with pytest.raises(NotImplementedError, match="no prompt-logprob support"):
        backend.score_prompt("anything", top_k=5)


def test_openai_compat_capabilities_reflect_provider(monkeypatch) -> None:
    backend_a, _ = _make_oc_backend(
        monkeypatch,
        routes={},
        has_completions=True,
        supports_prompt_logprobs=True,
        max_top=5,
    )
    backend_b, _ = _make_oc_backend(
        monkeypatch,
        routes={},
        has_completions=False,
        supports_prompt_logprobs=False,
        max_top=20,
    )

    assert backend_a.capabilities.prompt_logprobs is True
    assert backend_a.capabilities.can_force_token is True  # /completions
    assert backend_b.capabilities.prompt_logprobs is False
    assert backend_b.capabilities.can_force_token is False  # chat-only
    assert backend_b.capabilities.max_top_logprobs == 20


def test_openai_compat_intern_assigns_stable_ids(monkeypatch) -> None:
    backend, _ = _make_oc_backend(monkeypatch, routes={})

    a = backend._intern(" Paris")
    b = backend._intern(" London")
    c = backend._intern(" Paris")

    assert a != b
    assert a == c
    assert backend.piece(a) == " Paris"
    assert backend.detokenize([a, b]) == " Paris London"


def test_openai_compat_close_propagates(monkeypatch) -> None:
    backend, mock = _make_oc_backend(monkeypatch, routes={})

    backend.close()
    assert mock.closed is True


# --------------------------------------------------------------------------- #
# OpenAICompatBackend: 429/Retry-After / backoff
# --------------------------------------------------------------------------- #
def test_request_retries_on_429_and_honors_retry_after(monkeypatch) -> None:
    """A 429 with ``Retry-After`` is retried after the exact reported wait.

    This is the exact scenario that bit Fireworks/glm-5p2: the per-step
    decode loop burst past the per-account RPS limit, the server replied
    ``429`` with a small ``Retry-After``, and we used to surface that as
    an httpx exception terminating the whole generate stream. Now the
    backend swallows it and re-issues, so the user sees the generation
    complete (a touch slower) instead of an error toast.
    """
    sleeps: list[float] = []
    success_payload = {
        "choices": [
            {"logprobs": {"top_logprobs": [{" Paris": -0.5, " London": -2.0}]}}
        ]
    }
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/completions"): [
                (429, {"error": "throttled"}, {"Retry-After": "0.42"}),
                (200, success_payload, {}),
            ]
        },
        has_completions=True,
        max_retries=2,
        sleeps=sleeps,
    )

    step = backend.next_distribution(backend.tokenize("hi"), top_k=2)

    assert sleeps == [pytest.approx(0.42)]
    assert [c.text for c in step.candidates] == [" Paris", " London"]
    # The retry counted as a second POST (same URL) recorded in mock.calls.
    posts = [c for c in mock.calls if c["url"] == "/completions"]
    assert len(posts) == 2


def test_request_uses_exponential_backoff_when_no_retry_after(monkeypatch) -> None:
    """No ``Retry-After`` -> we fall back to base * 2**attempt + jitter.

    We patch ``random.uniform`` to 0 so the wait values are exact: with
    ``base_backoff_s=1.0`` and three retries, the first two waits are
    1.0s and 2.0s (the third attempt succeeds before sleeping again).
    """
    sleeps: list[float] = []
    monkeypatch.setattr(oc_mod.random, "uniform", lambda _a, _b: 0.0)
    success = {"choices": [{"logprobs": {"top_logprobs": [{"x": -0.1}]}}]}
    backend, _mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/completions"): [
                (429, {}, {}),
                (503, {}, {}),
                (200, success, {}),
            ]
        },
        has_completions=True,
        max_retries=3,
        sleeps=sleeps,
    )

    backend.next_distribution(backend.tokenize("hi"), top_k=1)

    assert sleeps == [1.0, 2.0]  # 1*2**0=1, 1*2**1=2


def test_request_exhausts_retries_and_raises(monkeypatch) -> None:
    """When every retry returns 429, the final ``raise_for_status`` fires.

    The MockResponse raises ``RuntimeError`` for non-2xx; the real
    backend raises ``httpx.HTTPStatusError``. Either way the caller
    (web/streaming) sees a propagated exception and emits a terminal
    ``done.error`` event -- matching the historical pre-fix behaviour
    for the genuine "we really are over quota" case.
    """
    sleeps: list[float] = []
    monkeypatch.setattr(oc_mod.random, "uniform", lambda _a, _b: 0.0)
    backend, _mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/completions"): [
                (429, {}, {"Retry-After": "0.01"}),
                (429, {}, {"Retry-After": "0.01"}),
                (429, {}, {"Retry-After": "0.01"}),
                (429, {}, {"Retry-After": "0.01"}),
            ]
        },
        has_completions=True,
        max_retries=2,
        sleeps=sleeps,
    )

    with pytest.raises(RuntimeError, match="HTTP 429"):
        backend.next_distribution(backend.tokenize("hi"), top_k=1)
    # Initial attempt + 2 retries = 3 total POSTs, 2 sleeps.
    assert len(sleeps) == 2


def test_request_does_not_retry_4xx_other_than_429(monkeypatch) -> None:
    """A 400/401/403/404 should NOT be retried -- the caller needs the body.

    Common cause: an unknown model name. Retrying just wastes seconds
    before the user sees the actual reason.
    """
    sleeps: list[float] = []
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={("POST", "/completions"): [(401, {"error": "bad key"}, {})]},
        has_completions=True,
        max_retries=5,
        sleeps=sleeps,
    )

    with pytest.raises(RuntimeError, match="HTTP 401"):
        backend.next_distribution(backend.tokenize("hi"), top_k=1)
    posts = [c for c in mock.calls if c["url"] == "/completions"]
    assert len(posts) == 1
    assert sleeps == []


# --------------------------------------------------------------------------- #
# OpenAICompatBackend: native streaming via /completions
# --------------------------------------------------------------------------- #
def _sse_lines(chunks: list[dict]) -> list[bytes]:
    """Turn JSON chunks into the line-stream httpx's iter_lines yields."""
    out: list[bytes] = []
    for ch in chunks:
        out.append(f"data: {json.dumps(ch)}".encode("utf-8"))
        out.append(b"")
    out.append(b"data: [DONE]")
    return out


class _MockStreamResponse:
    """Stand-in for the response yielded by ``httpx.Client.stream(...)``."""

    def __init__(self, status_code: int, lines: list[bytes], headers: dict | None = None) -> None:
        self.status_code = status_code
        self.headers = dict(headers or {})
        self._lines = lines

    def iter_lines(self):
        yield from self._lines

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")


class _MockStreamCM:
    """Context manager that ``httpx.Client.stream`` semantics expect."""

    def __init__(self, response: _MockStreamResponse) -> None:
        self._response = response

    def __enter__(self) -> _MockStreamResponse:
        return self._response

    def __exit__(self, *_excinfo) -> None:
        return None


def _attach_stream_factory(mock: MockHTTPClient, queue: list[_MockStreamResponse]) -> None:
    """Give the MockHTTPClient a ``.stream(...)`` method that pops responses."""

    @contextmanager
    def _stream(method, url, **kwargs):
        mock.calls.append({"method": method, "url": url, "kwargs": kwargs, "stream": True})
        if not queue:
            raise AssertionError("MockHTTPClient.stream: no more queued responses")
        resp = queue.pop(0)
        yield resp

    mock.stream = _stream  # type: ignore[attr-defined]


def test_supports_native_sampler_matrix(monkeypatch) -> None:
    """Built-ins on /completions backends are mappable; typical/custom are not.

    Chat-only providers always fall back to per-step because the native
    path requires the raw text-completion endpoint to emit echo-style
    per-token logprobs.
    """
    comp, _ = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    chat, _ = _make_oc_backend(monkeypatch, routes={}, has_completions=False)

    assert comp.supports_native_sampler("greedy", {})
    assert comp.supports_native_sampler("temperature", {"temperature": 0.7})
    assert comp.supports_native_sampler("top_p", {"top_p": 0.9})
    assert comp.supports_native_sampler("top_k", {"top_k": 10})
    assert comp.supports_native_sampler("min_p", {"min_p": 0.05})

    assert not comp.supports_native_sampler("typical", {"typical_p": 0.95})
    assert not comp.supports_native_sampler("custom", {})

    # Chat-only path can't do native streaming even for greedy.
    assert not chat.supports_native_sampler("greedy", {})


def test_stream_native_emits_one_genstep_per_token(monkeypatch) -> None:
    """A 3-token streamed completion becomes 3 GenStep events, last one tagged.

    Asserts (a) the request body translates the sampler correctly,
    (b) every emitted token is wrapped in a candidate list pulled from
    that step's ``top_logprobs``, and (c) the terminal ``finish_reason``
    propagates as the stop_reason on the *last* GenStep only.
    """
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, max_top=5
    )
    chunks = [
        {
            "choices": [
                {
                    "text": " Paris",
                    "logprobs": {
                        "tokens": [" Paris"],
                        "token_logprobs": [-0.2],
                        "top_logprobs": [{" Paris": -0.2, " London": -1.5}],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "text": " is",
                    "logprobs": {
                        "tokens": [" is"],
                        "token_logprobs": [-0.4],
                        "top_logprobs": [{" is": -0.4, " was": -1.8}],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "text": " warm",
                    "logprobs": {
                        "tokens": [" warm"],
                        "token_logprobs": [-0.6],
                        "top_logprobs": [{" warm": -0.6, " cold": -2.1}],
                    },
                    "finish_reason": "length",
                }
            ]
        },
    ]
    _attach_stream_factory(
        mock, [_MockStreamResponse(200, _sse_lines(chunks))]
    )

    steps = list(
        backend.stream_native(
            "The capital of France",
            sampler_name="top_p",
            sampler_params={"top_p": 0.85, "temperature": 0.7},
            max_tokens=3,
            top_k=5,
            stop_ids=None,
        )
    )

    # Request body was a single POST with translated sampler params.
    assert len(mock.calls) == 1
    sent = mock.calls[0]
    assert sent["url"] == "/completions"
    body = sent["kwargs"]["json"]
    assert body["stream"] is True
    assert body["logprobs"] == 5
    assert body["max_tokens"] == 3
    assert body["top_p"] == pytest.approx(0.85)
    assert body["temperature"] == pytest.approx(0.7)

    # Three GenSteps, one per token, "length" -> "max_tokens" on the last.
    assert len(steps) == 3
    assert [gs.decision.token_text for gs in steps] == [" Paris", " is", " warm"]
    assert steps[0].stop_reason is None
    assert steps[1].stop_reason is None
    assert steps[2].stop_reason == "max_tokens"
    # The note records that sampling happened server-side.
    assert "server-side" in steps[0].decision.note
    # tokens_before grows by one each step (the synthetic id of the prior token).
    assert len(steps[0].tokens_before) + 2 == len(steps[2].tokens_before)


def test_stream_native_maps_stop_finish_reason_to_user_stop(monkeypatch) -> None:
    """``finish_reason=stop`` propagates as ``stop_reason='user_stop'``."""
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    chunks = [
        {
            "choices": [
                {
                    "text": "X",
                    "logprobs": {
                        "tokens": ["X"],
                        "token_logprobs": [-0.1],
                        "top_logprobs": [{"X": -0.1}],
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    ]
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines(chunks))])

    steps = list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            stop_ids=None,
        )
    )
    assert steps[-1].stop_reason == "user_stop"


def test_stream_native_translates_stop_ids_to_text(monkeypatch) -> None:
    """``stop_ids`` are looked up via ``piece`` and sent as the ``stop`` array.

    Cloud providers don't speak our synthetic integer ids; OpenAI's
    ``stop`` array is text-based, so we translate. Capped at 4 entries
    to satisfy the OpenAI spec (some providers 400 on more than that).
    """
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    tid_a = backend._intern(" END")
    tid_b = backend._intern("\n\n")
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            stop_ids=[tid_a, tid_b],
        )
    )
    body = mock.calls[0]["kwargs"]["json"]
    assert body["stop"] == [" END", "\n\n"]


def test_stream_native_refuses_when_no_completions(monkeypatch) -> None:
    """Chat-only providers raise rather than silently picking a wrong path.

    The wire decision lives in
    :func:`decoding_sandbox.web.streaming._can_use_native_cloud_stream`;
    this test makes sure that even if someone calls the method directly
    with a chat-only provider, they get a loud :class:`NotImplementedError`
    they can branch on.
    """
    backend, _mock = _make_oc_backend(monkeypatch, routes={}, has_completions=False)
    with pytest.raises(NotImplementedError, match="no /completions endpoint"):
        list(
            backend.stream_native(
                "p",
                sampler_name="greedy",
                sampler_params={},
                max_tokens=1,
                top_k=1,
            )
        )


def test_stream_native_retries_initial_429(monkeypatch) -> None:
    """A 429 on the *initial* SSE open is retried using exp-backoff.

    Once bytes start flowing we don't retry mid-stream (the partial
    output would be wrong to retry), but the opening response code goes
    through the same retry path the JSON helper does. ``Retry-After``
    is honored just like for non-streaming endpoints.
    """
    sleeps: list[float] = []
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={},
        has_completions=True,
        max_retries=2,
        sleeps=sleeps,
    )
    chunks_after_retry = [
        {
            "choices": [
                {
                    "text": "T",
                    "logprobs": {
                        "tokens": ["T"],
                        "token_logprobs": [-0.1],
                        "top_logprobs": [{"T": -0.1}],
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    ]
    _attach_stream_factory(
        mock,
        [
            _MockStreamResponse(429, [], headers={"Retry-After": "0.2"}),
            _MockStreamResponse(200, _sse_lines(chunks_after_retry)),
        ],
    )

    steps = list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
        )
    )
    assert len(steps) == 1
    assert sleeps == [pytest.approx(0.2)]


# Smoke-test: MockResponse honors its new headers kwarg without breaking
# anything that used the legacy two-arg constructor.
def test_mock_response_headers_default_empty() -> None:
    r = MockResponse(200, {"ok": True})
    assert r.headers == {}
    r2 = MockResponse(429, {}, headers={"Retry-After": "1"})
    assert r2.headers["Retry-After"] == "1"


# --------------------------------------------------------------------------- #
# OpenAICompatBackend: seed / respect_eos / usage accounting
# --------------------------------------------------------------------------- #
def test_stream_native_body_carries_seed_and_include_usage(monkeypatch) -> None:
    """``seed`` and ``stream_options.include_usage`` are wired into the body.

    Both used to be silently dropped on the native cloud path: the UI
    collected a seed value, ``GenerateRequest`` validated it, the web
    layer passed it to ``stream_generate`` -- and the openai-compat
    backend never put it on the wire. ``include_usage`` is new: it's
    what makes the provider return the final ``usage`` chunk we now
    surface to the user as the ``usage`` SSE event.
    """
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])

    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            stop_ids=None,
            seed=42,
            respect_eos=True,
        )
    )

    body = mock.calls[-1]["kwargs"]["json"]
    assert body["seed"] == 42
    assert body["stream_options"] == {"include_usage": True}


def test_stream_native_respect_eos_false_adds_advisory_note(monkeypatch) -> None:
    """Cloud providers can't honor ``respect_eos=False``, so we say so.

    There is no documented OpenAI-compat field that asks the server
    to keep generating past EOS; every provider halts on the model's
    EOS token. Rather than silently lying about the flag, we leave a
    one-line note on the active usage sink so the UI can surface it
    next to the request counter.
    """
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    sink = {"requests": 0, "notes": []}
    backend.set_active_usage(sink)

    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            respect_eos=False,
        )
    )

    assert sink["notes"]
    assert any("respect_eos" in n for n in sink["notes"])


def test_request_counts_each_attempt_into_active_usage(monkeypatch) -> None:
    """``_request`` writes one to ``requests`` per HTTP attempt, retries included.

    The semantics we want for the UI is "how much did this run press
    on the provider's RPS budget". A 429-then-200 retry shows up as
    ``requests=2`` here, matching what the upstream load-balancer
    counted; the user can then visually correlate "weird latency" with
    "the backend retried this run twice".
    """
    sleeps: list[float] = []
    success_payload = {
        "choices": [
            {"logprobs": {"top_logprobs": [{" Paris": -0.5, " London": -2.0}]}}
        ]
    }
    backend, _mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/completions"): [
                (429, {}, {"Retry-After": "0.01"}),
                (200, success_payload, {}),
            ]
        },
        has_completions=True,
        max_retries=2,
        sleeps=sleeps,
    )
    sink = {"requests": 0}
    backend.set_active_usage(sink)

    backend.next_distribution(backend.tokenize("hi"), top_k=2)

    # Initial 429 + successful retry = 2 attempts = 2 increments.
    assert sink["requests"] == 2


def test_post_records_provider_reported_token_usage(monkeypatch) -> None:
    """A ``usage`` block in the JSON response feeds the active sink.

    Most OpenAI-compat providers attach a ``usage`` object to every
    completion (counts they bill against). Surfacing it lets the UI
    display real provider-side prompt/completion token counts instead
    of our best-effort local estimate.
    """
    payload = {
        "choices": [
            {"logprobs": {"top_logprobs": [{" hi": -0.1}]}}
        ],
        "usage": {"prompt_tokens": 13, "completion_tokens": 1, "total_tokens": 14},
    }
    backend, _mock = _make_oc_backend(
        monkeypatch,
        routes={("POST", "/completions"): payload},
        has_completions=True,
    )
    sink = {"requests": 0, "prompt_tokens": None, "completion_tokens": None}
    backend.set_active_usage(sink)

    backend.next_distribution(backend.tokenize("p"), top_k=1)

    assert sink["prompt_tokens"] == 13
    assert sink["completion_tokens"] == 1
    assert sink["total_tokens"] == 14


def test_stream_native_records_usage_from_final_chunk(monkeypatch) -> None:
    """The terminal SSE chunk's ``usage`` object lands in the active sink.

    With ``stream_options.include_usage=True`` upstream providers send
    a final chunk that has an empty ``choices`` array but a populated
    ``usage`` block. ``stream_native`` reads that out of band and
    forwards it via :func:`record_tokens` -- the per-token loop must
    NOT try to interpret that chunk as another emitted token.
    """
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    chunks = [
        {
            "choices": [
                {
                    "text": "X",
                    "logprobs": {
                        "tokens": ["X"],
                        "token_logprobs": [-0.05],
                        "top_logprobs": [{"X": -0.05}],
                    },
                    "finish_reason": "length",
                }
            ]
        },
        # Final SSE chunk that ``include_usage`` triggers.
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}},
    ]
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines(chunks))])
    sink = {"requests": 0, "prompt_tokens": None, "completion_tokens": None}
    backend.set_active_usage(sink)

    steps = list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
        )
    )

    assert len(steps) == 1  # the empty-choices usage chunk did NOT become a step
    assert steps[0].decision.token_text == "X"
    assert sink["prompt_tokens"] == 5
    assert sink["completion_tokens"] == 1
    assert sink["total_tokens"] == 6


def test_set_active_usage_clear_stops_accounting(monkeypatch) -> None:
    """Once the caller clears the sink, subsequent calls don't accrete onto it.

    The web layer clears ``backend.set_active_usage(None)`` in its
    ``finally`` so a later request can't see counters from a previous
    run. Reading from a cleared backend must be a clean no-op.
    """
    backend, _mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/completions"): {
                "choices": [{"logprobs": {"top_logprobs": [{" hi": -0.1}]}}]
            }
        },
        has_completions=True,
    )
    sink = {"requests": 0}
    backend.set_active_usage(sink)
    backend.next_distribution(backend.tokenize("p"), top_k=1)
    assert sink["requests"] == 1

    # Clear and run again -- the sink must stay at 1.
    backend.set_active_usage(None)
    backend.next_distribution(backend.tokenize("p"), top_k=1)
    assert sink["requests"] == 1


# --------------------------------------------------------------------------- #
# LlamaCppBackend
# --------------------------------------------------------------------------- #
def _make_llamacpp_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    routes: dict[tuple[str, str], object],
) -> tuple[LlamaCppBackend, MockHTTPClient]:
    mock = MockHTTPClient(routes)
    monkeypatch.setattr(llamacpp_mod.httpx, "Client", lambda **kw: mock)
    backend = LlamaCppBackend(base_url="http://server:8080")
    return backend, mock


def test_llamacpp_constructor_fetches_model_name(monkeypatch) -> None:
    backend, mock = _make_llamacpp_backend(
        monkeypatch,
        routes={("GET", "/v1/models"): {"data": [{"id": "qwen3-base"}]}},
    )
    assert backend.capabilities.name.endswith("qwen3-base")


def test_llamacpp_constructor_tolerates_missing_models_endpoint(monkeypatch) -> None:
    mock = MockHTTPClient({})  # no /v1/models registered
    monkeypatch.setattr(llamacpp_mod.httpx, "Client", lambda **kw: mock)
    backend = LlamaCppBackend(base_url="http://server:8080")
    assert backend.capabilities.name == "llamacpp:llama.cpp"


def test_llamacpp_tokenize_accepts_int_list_form(monkeypatch) -> None:
    backend, _ = _make_llamacpp_backend(
        monkeypatch,
        routes={
            ("GET", "/v1/models"): {"data": [{"id": "m"}]},
            ("POST", "/tokenize"): {"tokens": [42, 43, 44]},
        },
    )
    assert backend.tokenize("hi") == [42, 43, 44]


def test_llamacpp_tokenize_accepts_dict_form(monkeypatch) -> None:
    backend, _ = _make_llamacpp_backend(
        monkeypatch,
        routes={
            ("GET", "/v1/models"): {"data": [{"id": "m"}]},
            ("POST", "/tokenize"): {"tokens": [{"id": 1}, {"id": 2}]},
        },
    )
    assert backend.tokenize("hi") == [1, 2]


def test_llamacpp_detokenize_returns_content_field(monkeypatch) -> None:
    backend, _ = _make_llamacpp_backend(
        monkeypatch,
        routes={
            ("GET", "/v1/models"): {"data": [{"id": "m"}]},
            ("POST", "/detokenize"): {"content": "hello world"},
        },
    )
    assert backend.detokenize([1, 2, 3]) == "hello world"


def test_llamacpp_next_distribution_parses_n_probs(monkeypatch) -> None:
    backend, mock = _make_llamacpp_backend(
        monkeypatch,
        routes={
            ("GET", "/v1/models"): {"data": [{"id": "m"}]},
            ("POST", "/completion"): {
                "completion_probabilities": [
                    {
                        "top_logprobs": [
                            {"id": 100, "token": " Paris", "logprob": -0.4},
                            {"id": 200, "token": " London", "logprob": -2.3},
                        ]
                    }
                ]
            },
        },
    )

    step = backend.next_distribution([1, 2, 3], top_k=2)

    assert [c.text for c in step.candidates] == [" Paris", " London"]
    assert step.candidates[0].token_id == 100
    assert step.candidates[0].rank == 0
    assert step.is_full_vocab is False
    assert step.position == 3
    # Server received the right body:
    body = mock.calls[-1]["json"]
    assert body["prompt"] == [1, 2, 3]
    assert body["n_probs"] == 2
    assert body["temperature"] == 0.0
    assert body["cache_prompt"] is True


def test_llamacpp_next_distribution_returns_empty_when_server_returns_nothing(monkeypatch) -> None:
    backend, _ = _make_llamacpp_backend(
        monkeypatch,
        routes={
            ("GET", "/v1/models"): {"data": [{"id": "m"}]},
            ("POST", "/completion"): {"completion_probabilities": []},
        },
    )
    step = backend.next_distribution([1, 2], top_k=5)
    assert step.candidates == []
    assert step.position == 2


def test_llamacpp_clamps_top_k_to_max_top(monkeypatch) -> None:
    backend, mock = _make_llamacpp_backend(
        monkeypatch,
        routes={
            ("GET", "/v1/models"): {"data": [{"id": "m"}]},
            ("POST", "/completion"): {"completion_probabilities": [{"top_logprobs": []}]},
        },
    )

    backend.next_distribution([1], top_k=10_000)
    body = mock.calls[-1]["json"]
    assert body["n_probs"] == backend._max_top  # default 40


def test_llamacpp_piece_caches(monkeypatch) -> None:
    backend, mock = _make_llamacpp_backend(
        monkeypatch,
        routes={
            ("GET", "/v1/models"): {"data": [{"id": "m"}]},
            ("POST", "/detokenize"): {"content": "X"},
        },
    )

    a = backend.piece(7)
    b = backend.piece(7)

    detokenize_calls = [c for c in mock.calls if c["url"] == "/detokenize"]
    assert a == b == "X"
    assert len(detokenize_calls) == 1


def test_llamacpp_close_propagates(monkeypatch) -> None:
    backend, mock = _make_llamacpp_backend(
        monkeypatch,
        routes={("GET", "/v1/models"): {"data": [{"id": "m"}]}},
    )
    backend.close()
    assert mock.closed is True


def test_llamacpp_capabilities_advertise_top_k_only(monkeypatch) -> None:
    backend, _ = _make_llamacpp_backend(
        monkeypatch,
        routes={("GET", "/v1/models"): {"data": [{"id": "m"}]}},
    )
    caps = backend.capabilities
    assert caps.full_vocab is False
    assert caps.prompt_logprobs is False
    assert caps.can_force_token is True
    assert caps.max_top_logprobs == 40


def test_llamacpp_candidates_have_correct_logprob_math(monkeypatch) -> None:
    """sanity-check: the probability matches exp(logprob)."""
    backend, _ = _make_llamacpp_backend(
        monkeypatch,
        routes={
            ("GET", "/v1/models"): {"data": [{"id": "m"}]},
            ("POST", "/completion"): {
                "completion_probabilities": [
                    {"top_logprobs": [{"id": 1, "token": "a", "logprob": math.log(0.6)}]}
                ]
            },
        },
    )
    step = backend.next_distribution([0], top_k=1)
    assert step.candidates[0].prob == pytest.approx(0.6)
