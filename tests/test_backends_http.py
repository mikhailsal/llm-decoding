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
    supports_ignore_eos: bool = False,
    supports_perf_metrics: bool = False,
    supports_service_tier: bool = False,
    supports_prompt_cache_key: bool = False,
    supports_session_affinity: bool = False,
    supports_new_logprobs: bool = False,
    supports_sampling_mask: bool = False,
    supports_raw_output: bool = False,
    supports_logit_bias: bool = False,
    supports_combined_echo_stream: bool = False,
) -> tuple[OpenAICompatBackend, MockHTTPClient]:
    """Build an OpenAICompatBackend wired to a MockHTTPClient.

    ``max_retries`` defaults to 0 so existing tests stay deterministic
    (one shot, raises on non-2xx). Tests that exercise retry pass it
    explicitly along with a ``sleeps`` list that captures wait values
    instead of actually sleeping.

    ``supports_*`` mirror :class:`ProviderConfig` extension flags so each
    test can opt in to a Fireworks-style provider without affecting the
    default conservative profile (no extensions) other tests rely on.
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
        supports_ignore_eos=supports_ignore_eos,
        supports_perf_metrics=supports_perf_metrics,
        supports_service_tier=supports_service_tier,
        supports_prompt_cache_key=supports_prompt_cache_key,
        supports_session_affinity=supports_session_affinity,
        supports_new_logprobs=supports_new_logprobs,
        supports_sampling_mask=supports_sampling_mask,
        supports_raw_output=supports_raw_output,
        supports_logit_bias=supports_logit_bias,
        supports_combined_echo_stream=supports_combined_echo_stream,
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


def test_openai_compat_new_logprobs_request_shape(monkeypatch) -> None:
    """``supports_new_logprobs`` flips the body from ``logprobs: N`` to ``true`` + ``top_logprobs``.

    Pinning the wire shape here is the cheapest possible regression
    guard: an accidental refactor that re-collapses the two paths
    would silently downgrade Fireworks to the legacy format and lose
    real token_ids / sampling_mask_count -- both invisible until a
    user fires up ``--watch-id`` and gets nothing.
    """
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/completions"): {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": " Paris",
                                    "token_id": 12345,
                                    "logprob": -0.5,
                                    "sampling_mask_count": 42,
                                    "top_logprobs": [
                                        {"token": " Paris", "token_id": 12345, "logprob": -0.5},
                                        {"token": " London", "token_id": 67890, "logprob": -2.0},
                                    ],
                                }
                            ]
                        }
                    }
                ]
            }
        },
        has_completions=True,
        supports_new_logprobs=True,
        supports_sampling_mask=True,
    )
    step = backend.next_distribution(backend.tokenize("hi"), top_k=2)
    body = mock.calls[-1]["json"]
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 2
    assert body["sampling_mask"] == "count"
    # Real token IDs from the response, not synthetic interns.
    assert step.candidates[0].token_id == 12345
    assert step.candidates[1].token_id == 67890
    # sampling_mask_count was stamped onto every candidate at this position.
    assert step.candidates[0].sampling_mask_count == 42
    assert step.candidates[1].sampling_mask_count == 42


def test_openai_compat_legacy_logprobs_request_shape_when_disabled(monkeypatch) -> None:
    """Without ``supports_new_logprobs`` we still ship the integer form."""
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/completions"): {
                "choices": [
                    {"logprobs": {"top_logprobs": [{" Paris": -0.5}]}}
                ]
            }
        },
        has_completions=True,
    )
    backend.next_distribution(backend.tokenize("hi"), top_k=3)
    body = mock.calls[-1]["json"]
    assert body["logprobs"] == 3
    assert "top_logprobs" not in body
    assert "sampling_mask" not in body


def test_openai_compat_intern_ids_dont_collide_with_real_token_ids(monkeypatch) -> None:
    """Synthetic intern ids are namespaced above real model ids.

    Real ids returned by NewLogProbs sit in the [0, vocab_size) range
    (Fireworks models top out around ~256K). Synthetic ids from
    ``_intern`` are offset by ``_INTERN_ID_BASE`` (≥16M) so the two
    spaces never overlap; otherwise a ``tokenize(prompt)`` call could
    silently overwrite ``_id_to_text[42]`` after a real token 42 had
    landed there from a previous score_prompt.
    """
    backend, _ = _make_oc_backend(monkeypatch, routes={})
    a = backend._intern("hello")
    b = backend._intern("world")
    assert a >= backend._INTERN_ID_BASE
    assert b >= backend._INTERN_ID_BASE


def test_openai_compat_score_prompt_new_logprobs_uses_real_ids(monkeypatch) -> None:
    """NewLogProbs echo carries real prompt token ids + sampling_mask_count.

    The chosen field for each real-prompt position must point at the
    SAME id reported in the position's top_logprobs entry (rank 0 for
    a deterministic max_tokens=1, temperature=0 score). Without
    NewLogProbs the backend used to invent synthetic ids that would
    never match watch ids derived from the model's actual vocab.
    """
    backend, _ = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/completions"): {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": "The",
                                    "token_id": 100,
                                    "logprob": -2.0,
                                    "sampling_mask_count": 10,
                                    "top_logprobs": [
                                        {"token": "The", "token_id": 100, "logprob": -2.0}
                                    ],
                                },
                                {
                                    "token": " cap",
                                    "token_id": 200,
                                    "logprob": -1.0,
                                    "sampling_mask_count": 8,
                                    "top_logprobs": [
                                        {"token": " cap", "token_id": 200, "logprob": -1.0},
                                        {"token": " other", "token_id": 999, "logprob": -3.0},
                                    ],
                                },
                                {
                                    "token": " city",
                                    "token_id": 300,
                                    "logprob": -0.5,
                                    "sampling_mask_count": 50,
                                    "top_logprobs": [
                                        {"token": " city", "token_id": 300, "logprob": -0.5},
                                        {"token": " is", "token_id": 301, "logprob": -1.5},
                                    ],
                                },
                            ]
                        }
                    }
                ]
            }
        },
        has_completions=True,
        supports_prompt_logprobs=True,
        supports_new_logprobs=True,
        supports_sampling_mask=True,
    )
    steps = backend.score_prompt("The capital", top_k=3, watch_ids=[])
    assert len(steps) == 2
    real_prompt_step, predict_next = steps
    # Real prompt position: chosen carries the REAL id from the response.
    assert real_prompt_step.chosen is not None
    assert real_prompt_step.chosen.token_id == 200
    assert real_prompt_step.chosen.sampling_mask_count == 8
    # Predict-next trailing slot: chosen=None as always; mask_count
    # still stamped on candidates so the UI can render the column.
    assert predict_next.chosen is None
    assert predict_next.candidates[0].sampling_mask_count == 50


def test_openai_compat_stream_native_new_logprobs_parses_content(monkeypatch) -> None:
    """SSE chunks carry ``logprobs.content[]`` instead of parallel arrays."""
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={},
        has_completions=True,
        supports_new_logprobs=True,
        supports_sampling_mask=True,
    )
    chunks = [
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": " Paris",
                                "token_id": 12345,
                                "logprob": -0.2,
                                "sampling_mask_count": 7,
                                "top_logprobs": [
                                    {"token": " Paris", "token_id": 12345, "logprob": -0.2},
                                    {"token": " London", "token_id": 67890, "logprob": -1.5},
                                ],
                            }
                        ]
                    },
                    "finish_reason": "length",
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
            top_k=2,
        )
    )
    assert len(steps) == 1
    gs = steps[0]
    body = mock.calls[-1]["kwargs"]["json"]
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 2
    assert body["sampling_mask"] == "count"
    # The emitted token carries the real model id, not an intern.
    assert gs.decision.token_id == 12345
    assert gs.step_result.chosen.token_id == 12345
    # sampling_mask_count appears on every candidate at this step.
    assert gs.step_result.candidates[0].sampling_mask_count == 7
    # tokens_before grew by one real id; previous ids are the prompt
    # intern (we don't have a real tokenizer for the prompt, so the
    # synthetic intern id is fine -- but it must be in the intern
    # range, NOT overlapping the real id 12345).
    assert all(
        t >= backend._INTERN_ID_BASE for t in gs.tokens_before
    ), gs.tokens_before


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


def test_openai_compat_capabilities_surface_extension_flags(monkeypatch) -> None:
    """ProviderConfig.supports_* extension flags appear in Capabilities.

    The frontend reads ``Capabilities.supports_ignore_eos`` to decide
    whether to lock the ``respect EOS`` checkbox; if the surface here
    silently swallowed the flag the UI would default back to "always
    locked" for Fireworks too -- the exact bug the audit flagged.
    """
    fw, _ = _make_oc_backend(
        monkeypatch,
        routes={},
        has_completions=True,
        supports_ignore_eos=True,
        supports_perf_metrics=True,
        supports_service_tier=True,
    )
    plain, _ = _make_oc_backend(monkeypatch, routes={}, has_completions=False)

    assert fw.capabilities.supports_ignore_eos is True
    assert fw.capabilities.supports_perf_metrics is True
    assert fw.capabilities.supports_service_tier is True
    assert plain.capabilities.supports_ignore_eos is False
    assert plain.capabilities.supports_perf_metrics is False
    assert plain.capabilities.supports_service_tier is False


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
    """Built-ins are universally mappable; typical/mirostat gated per-provider.

    Chat-only providers always fall back to per-step because the native
    path requires the raw text-completion endpoint to emit echo-style
    per-token logprobs. ``typical_p`` and ``mirostat`` are
    Fireworks-extensions so they only count as native when the provider
    explicitly opts in -- otherwise the per-step fallback runs the
    local implementation (mirostat v2 + typical filters in
    :mod:`decoding_sandbox.core.samplers`).
    """
    comp, _ = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    chat, _ = _make_oc_backend(monkeypatch, routes={}, has_completions=False)

    assert comp.supports_native_sampler("greedy", {})
    assert comp.supports_native_sampler("temperature", {"temperature": 0.7})
    assert comp.supports_native_sampler("top_p", {"top_p": 0.9})
    assert comp.supports_native_sampler("top_k", {"top_k": 10})
    assert comp.supports_native_sampler("min_p", {"min_p": 0.05})

    # Default profile: no extension flags -> typical / mirostat are NOT native.
    assert not comp.supports_native_sampler("typical", {"typical_p": 0.95})
    assert not comp.supports_native_sampler("mirostat", {"mirostat_target": 5.0})
    assert not comp.supports_native_sampler("custom", {})

    # Fireworks-style profile: extensions on -> both are native.
    fw, _ = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True,
    )
    fw.provider.supports_typical_p_native = True
    fw.provider.supports_mirostat = True
    assert fw.supports_native_sampler("typical", {"typical_p": 0.95})
    assert fw.supports_native_sampler("mirostat", {"mirostat_target": 5.0})

    # Chat-only path can't do native streaming even for greedy.
    assert not chat.supports_native_sampler("greedy", {})


def test_stream_native_forwards_typical_p_when_supported(monkeypatch) -> None:
    """Fireworks-extension typical_p flows into the request body."""
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    backend.provider.supports_typical_p_native = True
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="typical",
            sampler_params={"typical_p": 0.7, "temperature": 0.6},
            max_tokens=1,
            top_k=1,
        )
    )
    body = mock.calls[-1]["kwargs"]["json"]
    assert body["typical_p"] == pytest.approx(0.7)
    assert body["temperature"] == pytest.approx(0.6)


def test_stream_native_forwards_mirostat_target_and_lr(monkeypatch) -> None:
    """mirostat_target + mirostat_lr land on the wire as plain body keys."""
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    backend.provider.supports_mirostat = True
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="mirostat",
            sampler_params={"mirostat_target": 3.5, "mirostat_lr": 0.2, "temperature": 0.9},
            max_tokens=1,
            top_k=1,
        )
    )
    body = mock.calls[-1]["kwargs"]["json"]
    assert body["mirostat_target"] == pytest.approx(3.5)
    assert body["mirostat_lr"] == pytest.approx(0.2)
    assert body["temperature"] == pytest.approx(0.9)


def test_stream_native_forwards_frequency_and_presence_penalties(monkeypatch) -> None:
    """``frequency_penalty`` / ``presence_penalty`` are standard OpenAI fields.

    We ship them unconditionally (no provider-extension gate) BUT only
    when their value differs from the no-op default. That way a user
    who never touched the inputs doesn't see them on the wire even on
    providers that accept them; a user who set freq=0.5 sees the field
    appear regardless of provider.
    """
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="top_p",
            sampler_params={"top_p": 0.9, "frequency_penalty": 0.5, "presence_penalty": 0.3},
            max_tokens=1,
            top_k=1,
        )
    )
    body = mock.calls[-1]["kwargs"]["json"]
    assert body["frequency_penalty"] == pytest.approx(0.5)
    assert body["presence_penalty"] == pytest.approx(0.3)


def test_stream_native_repetition_penalty_gated_by_provider(monkeypatch) -> None:
    """``repetition_penalty`` is Fireworks-only; non-supporting providers drop it."""
    plain, mock_a = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    _attach_stream_factory(mock_a, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        plain.stream_native(
            "p",
            sampler_name="top_p",
            sampler_params={"top_p": 0.9, "repetition_penalty": 1.1},
            max_tokens=1,
            top_k=1,
        )
    )
    body_a = mock_a.calls[-1]["kwargs"]["json"]
    assert "repetition_penalty" not in body_a

    fw, mock_b = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    fw.provider.supports_repetition_penalty = True
    _attach_stream_factory(mock_b, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        fw.stream_native(
            "p",
            sampler_name="top_p",
            sampler_params={"top_p": 0.9, "repetition_penalty": 1.1},
            max_tokens=1,
            top_k=1,
        )
    )
    body_b = mock_b.calls[-1]["kwargs"]["json"]
    assert body_b["repetition_penalty"] == pytest.approx(1.1)


def test_stream_native_omits_no_op_penalty_defaults(monkeypatch) -> None:
    """Defaults (``freq=0``, ``pres=0``, ``rep=1.0``) MUST NOT appear on wire.

    Strict OpenAI-compat providers (or future-OpenAI itself) sometimes
    reject zero penalties; even when they accept, the empty bytes are
    just noise. Pin the default-shape so a regression doesn't quietly
    grow the wire body.
    """
    fw, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    fw.provider.supports_repetition_penalty = True
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        fw.stream_native(
            "p",
            sampler_name="top_p",
            sampler_params={"top_p": 0.9},
            max_tokens=1,
            top_k=1,
        )
    )
    body = mock.calls[-1]["kwargs"]["json"]
    assert "frequency_penalty" not in body
    assert "presence_penalty" not in body
    assert "repetition_penalty" not in body


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

    sink = {"requests": 0}
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
    assert len(steps) == 1
    assert sleeps == [pytest.approx(0.2)]
    # Both the 429 attempt and the successful retry must show up in the
    # request counter, matching how ``_request`` counts non-streaming
    # retries -- otherwise a user fighting RPS limits sees stable "1
    # request" even when we're retrying for them under the hood.
    assert sink["requests"] == 2


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
    """Cloud providers without ``ignore_eos`` get a one-line advisory note.

    There is no OpenAI-compat field that asks the server to keep
    generating past EOS, so any provider that hasn't opted in to the
    Fireworks-style ``ignore_eos`` extension just halts. Rather than
    silently lying about the flag, we leave a one-line note on the
    active usage sink so the UI can surface it next to the request
    counter. The test pins the default profile (no extensions) to
    document this fallback explicitly.
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
    # The body must NOT carry ``ignore_eos`` when the provider can't
    # honor it: shipping a field the upstream's request validator
    # rejects (or worse, silently misinterprets) is exactly the kind
    # of "silent lie" we want to avoid.
    body = mock.calls[-1]["kwargs"]["json"]
    assert "ignore_eos" not in body


def test_stream_native_respect_eos_false_fireworks_forwards_ignore_eos(monkeypatch) -> None:
    """On a Fireworks-style provider, ``respect_eos=False`` ships ``ignore_eos: true``.

    With ``supports_ignore_eos=True`` the backend is allowed to use the
    Fireworks extension that disables EOS halting server-side. The
    advisory note must NOT appear because the upstream actually honors
    the flag now -- showing it would falsely suggest the request was
    silently downgraded.
    """
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, supports_ignore_eos=True
    )
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

    body = mock.calls[-1]["kwargs"]["json"]
    assert body["ignore_eos"] is True
    # No advisory: the provider honored the flag for real this time.
    assert not any("respect_eos" in n for n in sink.get("notes", []))


def test_stream_native_respect_eos_true_omits_ignore_eos(monkeypatch) -> None:
    """``respect_eos=True`` must never set ``ignore_eos`` (even on Fireworks).

    The whole point of "respect EOS" is to let the model halt. Shipping
    ``ignore_eos: true`` regardless would erase that behaviour and make
    the default mode useless for anything but explicit non-stop sweeps.
    """
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, supports_ignore_eos=True
    )
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])

    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            respect_eos=True,
        )
    )

    body = mock.calls[-1]["kwargs"]["json"]
    assert "ignore_eos" not in body


def test_stream_native_requests_perf_metrics_when_supported(monkeypatch) -> None:
    """Fireworks-style providers always get ``perf_metrics_in_response: true``.

    Cheap to ship (small object), but turns the educational sandbox
    into a proper "where is the time going" debugger by exposing TTFT
    + prefill + generation timings. Providers without the flag must
    NOT see the field on the wire (some are strict about unknown
    body keys, and even when they aren't, sending dead bytes is
    wasteful).
    """
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, supports_perf_metrics=True
    )
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p", sampler_name="greedy", sampler_params={}, max_tokens=1, top_k=1
        )
    )
    assert mock.calls[-1]["kwargs"]["json"]["perf_metrics_in_response"] is True


def test_stream_native_omits_perf_metrics_when_unsupported(monkeypatch) -> None:
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p", sampler_name="greedy", sampler_params={}, max_tokens=1, top_k=1
        )
    )
    assert "perf_metrics_in_response" not in mock.calls[-1]["kwargs"]["json"]


def test_stream_native_records_perf_metrics_from_final_chunk(monkeypatch) -> None:
    """The ``perf_metrics`` block in the final SSE chunk lands on the sink.

    Streaming providers don't have a "response body" the way non-stream
    calls do -- the perf block comes in the *last* chunk (same chunk
    as the ``usage`` block when ``include_usage`` is on). The web
    layer reads it back from the sink and emits a dedicated ``perf``
    SSE frame to the browser.
    """
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, supports_perf_metrics=True
    )
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
                    "finish_reason": "length",
                }
            ]
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "perf_metrics": {
                "prompt-tokens": 1,
                "server-time-to-first-token": 0.042,
                "prefill-duration": 0.011,
                "generation-duration": 0.031,
            },
        },
    ]
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines(chunks))])
    sink: dict = {"requests": 0, "notes": [], "perf_metrics": None}
    backend.set_active_usage(sink)
    list(
        backend.stream_native(
            "p", sampler_name="greedy", sampler_params={}, max_tokens=1, top_k=1
        )
    )
    assert isinstance(sink["perf_metrics"], dict)
    assert sink["perf_metrics"]["server-time-to-first-token"] == pytest.approx(0.042)
    assert sink["perf_metrics"]["prompt-tokens"] == 1


def test_stream_native_records_perf_metrics_from_response_body(monkeypatch) -> None:
    """Non-stream JSON responses also surface ``perf_metrics`` to the sink.

    The ``_post`` path (used by next_distribution / score_prompt) reads
    the metrics from the response body directly. Without this code
    path the inspect page would never see server timings.
    """
    payload = {
        "choices": [
            {"logprobs": {"top_logprobs": [{" Paris": -0.5}]}}
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        "perf_metrics": {"server-processing-time": 0.123},
    }
    backend, _mock = _make_oc_backend(
        monkeypatch,
        routes={("POST", "/completions"): payload},
        has_completions=True,
        supports_perf_metrics=True,
    )
    sink: dict = {"requests": 0, "perf_metrics": None}
    backend.set_active_usage(sink)
    backend.next_distribution(backend.tokenize("hi"), top_k=1)
    assert sink["perf_metrics"] == {"server-processing-time": pytest.approx(0.123)}


def test_stream_native_requests_raw_output_when_supported(monkeypatch) -> None:
    """``raw_output: true`` is always on for providers that support it.

    The wire test pins the always-on behaviour because the only way
    the UI's "what the model saw" panel ever has anything to render
    is if every request asks for the diagnostics upfront -- the
    user can't decide retroactively to see what their last completion
    saw.
    """
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, supports_raw_output=True
    )
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p", sampler_name="greedy", sampler_params={}, max_tokens=1, top_k=1
        )
    )
    assert mock.calls[-1]["kwargs"]["json"]["raw_output"] is True


def test_stream_native_omits_raw_output_when_unsupported(monkeypatch) -> None:
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p", sampler_name="greedy", sampler_params={}, max_tokens=1, top_k=1
        )
    )
    assert "raw_output" not in mock.calls[-1]["kwargs"]["json"]


def test_stream_native_records_raw_output_from_final_chunk(monkeypatch) -> None:
    """The provider's ``raw_output`` block ends up on the usage sink.

    Stream version: the diagnostics arrive in the final ``include_usage``
    chunk alongside ``usage`` and ``perf_metrics``. We pin the parsing
    here because the UI's "what the model saw" panel reads
    ``sink["raw_output"]`` straight through.
    """
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, supports_raw_output=True
    )
    raw_payload = {
        "prompt_fragments": ["<|system|>", "you are useful", "<|user|>", "hi"],
        "prompt_token_ids": [1, 200, 201, 5, 300],
        "grammar": {"kind": "json_schema", "name": "noop"},
    }
    chunks = [
        {
            "choices": [{"finish_reason": "length"}],
            "raw_output": raw_payload,
        }
    ]
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines(chunks))])
    sink: dict = {"requests": 0}
    backend.set_active_usage(sink)
    list(
        backend.stream_native(
            "p", sampler_name="greedy", sampler_params={}, max_tokens=1, top_k=1
        )
    )
    assert sink["raw_output"] == raw_payload


def test_post_records_raw_output_from_response_body(monkeypatch) -> None:
    """Non-stream JSON path also surfaces ``raw_output`` to the sink."""
    payload = {
        "choices": [{"logprobs": {"top_logprobs": [{" Paris": -0.5}]}}],
        "raw_output": {"prompt_fragments": ["hi"]},
    }
    backend, _mock = _make_oc_backend(
        monkeypatch,
        routes={("POST", "/completions"): payload},
        has_completions=True,
        supports_raw_output=True,
    )
    sink: dict = {"requests": 0}
    backend.set_active_usage(sink)
    backend.next_distribution(backend.tokenize("hi"), top_k=1)
    assert sink["raw_output"] == {"prompt_fragments": ["hi"]}


def test_stream_native_forwards_logit_bias_when_supported(monkeypatch) -> None:
    """``logit_bias`` lands in the body as a stringified-keys dict.

    OpenAI Completions takes string keys for `logit_bias` (a JSON
    requirement); we coerce ints to strings at the wire boundary and
    drop NaN / out-of-range / non-numeric entries silently so a single
    bad key doesn't fail the whole request.
    """
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, supports_logit_bias=True
    )
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            logit_bias={123: 5.0, 456: -100.0, 789: 200.0, 0: float("nan")},
        )
    )
    body = mock.calls[-1]["kwargs"]["json"]
    # In-range entries pass through; out-of-range and NaN are filtered.
    assert body["logit_bias"] == {"123": 5.0, "456": -100.0}


def test_stream_native_drops_logit_bias_when_unsupported(monkeypatch) -> None:
    """Backends without the capability never see ``logit_bias`` on the wire."""
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            logit_bias={123: 5.0},
        )
    )
    assert "logit_bias" not in mock.calls[-1]["kwargs"]["json"]


def test_stream_native_omits_logit_bias_when_empty(monkeypatch) -> None:
    """An empty / all-filtered logit_bias map should not appear on the wire."""
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, supports_logit_bias=True
    )
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            logit_bias={0: float("nan"), 1: 999.0},
        )
    )
    assert "logit_bias" not in mock.calls[-1]["kwargs"]["json"]


def test_stream_native_with_echo_requires_capability(monkeypatch) -> None:
    """``stream_native_with_echo`` refuses to run on providers that haven't opted in.

    Without an explicit capability flag we'd risk silently sending
    ``echo=true`` + ``stream=true`` to a provider that 400s on the
    combo. Better to surface a clear NotImplementedError so the web
    layer's ``_can_use_combined_echo_stream`` check stays the single
    source of truth.
    """
    backend, _ = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    with pytest.raises(NotImplementedError, match="supports_combined_echo_stream"):
        list(
            backend.stream_native_with_echo(
                "p",
                sampler_name="greedy",
                sampler_params={},
                max_tokens=1,
                top_k=1,
            )
        )


def test_stream_native_with_echo_splits_prompt_and_generation(monkeypatch) -> None:
    """One streaming POST yields BOTH prompt-echo StepResults AND emitted GenSteps.

    Phase 5 payoff: instead of two HTTP requests (``score_prompt`` +
    ``stream_native``) the combined path makes one. We canon-pin the
    split heuristic here -- the first chunk's positions are prompt
    echo; later chunks contribute emitted tokens. If a future
    refactor breaks the split point detection, the prompt_score frame
    would silently lose rows (or include emitted tokens), which the
    UI table would render as confusing extra rows -- exactly the kind
    of regression a wire-shape test catches cheaply.
    """
    from decoding_sandbox.core.engine import GenStep
    from decoding_sandbox.core.types import StepResult

    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={},
        has_completions=True,
        supports_new_logprobs=True,
        supports_combined_echo_stream=True,
    )
    # Chunk #1: 3 echoed prompt positions in one batch.
    # Chunks #2 and #3: one emitted token each.
    chunks = [
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "The",
                                "token_id": 100,
                                "logprob": -2.0,
                                "top_logprobs": [{"token": "The", "token_id": 100, "logprob": -2.0}],
                            },
                            {
                                "token": " cap",
                                "token_id": 200,
                                "logprob": -1.0,
                                "top_logprobs": [{"token": " cap", "token_id": 200, "logprob": -1.0}],
                            },
                            {
                                "token": " of",
                                "token_id": 300,
                                "logprob": -0.5,
                                "top_logprobs": [{"token": " of", "token_id": 300, "logprob": -0.5}],
                            },
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": " France",
                                "token_id": 400,
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": " France", "token_id": 400, "logprob": -0.1}
                                ],
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": " is",
                                "token_id": 500,
                                "logprob": -0.05,
                                "top_logprobs": [
                                    {"token": " is", "token_id": 500, "logprob": -0.05}
                                ],
                            }
                        ]
                    },
                    "finish_reason": "length",
                }
            ]
        },
    ]
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines(chunks))])
    out = list(
        backend.stream_native_with_echo(
            "The cap of",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=2,
            top_k=1,
        )
    )
    body = mock.calls[-1]["kwargs"]["json"]
    assert body["echo"] is True
    assert body["stream"] is True
    # 3 prompt StepResults + 2 GenSteps = 5 yields total.
    step_results = [x for x in out if isinstance(x, StepResult)]
    gen_steps = [x for x in out if isinstance(x, GenStep)]
    assert len(step_results) == 3, [type(x).__name__ for x in out]
    assert len(gen_steps) == 2, [type(x).__name__ for x in out]
    # Prompt StepResults carry the right token ids.
    assert [s.chosen.token_id for s in step_results] == [100, 200, 300]
    # GenSteps mirror the emitted token sequence with their real ids.
    assert [g.decision.token_id for g in gen_steps] == [400, 500]
    # Stop reason landed on the LAST GenStep only.
    assert gen_steps[-1].stop_reason == "max_tokens"


def test_stream_native_with_echo_splits_by_text_offset(monkeypatch) -> None:
    """Real Fireworks streams one position per SSE chunk; the split
    must use ``text_offset`` (or fall back to cumulative length /
    sampling-signal presence) rather than chunk boundaries.

    The original implementation assumed "first chunk = all echo, rest
    = emit" -- which collapses to "1 echo + N emit" on real Fireworks
    where every position arrives in its own chunk. The Chrome MCP
    manual check caught this regression: a 4-token prompt rendered
    only 1 row in the prompt-logits table. This pin reproduces the
    exact wire shape (one entry per chunk, with text_offset, with
    sampling_logprob on emit positions only) so any future drift in
    the split heuristic fails loudly.
    """
    from decoding_sandbox.core.engine import GenStep
    from decoding_sandbox.core.types import StepResult

    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={},
        has_completions=True,
        supports_new_logprobs=True,
        supports_combined_echo_stream=True,
    )

    def _echo_entry(text: str, tid: int, lp: float, offset: int) -> dict:
        return {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": text,
                                "token_id": tid,
                                "logprob": lp,
                                "text_offset": offset,
                                "top_logprobs": [
                                    {"token": text, "token_id": tid, "logprob": lp}
                                ],
                            }
                        ]
                    }
                }
            ]
        }

    def _emit_entry(
        text: str, tid: int, lp: float, offset: int, finish: str | None = None
    ) -> dict:
        return {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": text,
                                "token_id": tid,
                                "logprob": lp,
                                "text_offset": offset,
                                "sampling_logprob": lp,
                                "sampling_mask_count": 100,
                                "top_logprobs": [
                                    {"token": text, "token_id": tid, "logprob": lp}
                                ],
                            }
                        ]
                    },
                    "finish_reason": finish,
                }
            ]
        }

    # Prompt "Hi friend" = 9 chars, tokens: ["Hi", " friend"] at offsets 0, 2.
    # Emit: "!" at offset 9, "." at offset 10.
    chunks = [
        _echo_entry("Hi", 100, 0.0, 0),
        _echo_entry(" friend", 200, -1.0, 2),
        _emit_entry("!", 300, -0.5, 9),
        _emit_entry(".", 400, -0.8, 10, finish="length"),
    ]
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines(chunks))])
    out = list(
        backend.stream_native_with_echo(
            "Hi friend",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=2,
            top_k=1,
        )
    )
    step_results = [x for x in out if isinstance(x, StepResult)]
    gen_steps = [x for x in out if isinstance(x, GenStep)]
    # The fix: BOTH echo positions land in prompt_score, BOTH emit
    # positions land in the gen stream. Pre-fix only the first chunk
    # contributed to prompt_score, so this would have asserted 1/3
    # instead of the correct 2/2.
    assert [s.chosen.token_id for s in step_results] == [100, 200]
    assert [g.decision.token_id for g in gen_steps] == [300, 400]
    # Sampling-mask data only on emit positions (Fireworks behavior).
    assert all(s.chosen.sampling_mask_count is None for s in step_results)
    assert all(
        g.step_result.chosen.sampling_mask_count == 100 for g in gen_steps
    )


def test_stream_native_with_echo_splits_by_sampling_signal(monkeypatch) -> None:
    """Fallback split signal: ``sampling_mask_count`` presence.

    When the provider omits ``text_offset`` (and the cumulative-text
    fallback doesn't help because tokens contain weird unicode), we
    fall back to "first entry that carries a non-null
    ``sampling_logprob`` / ``sampling_mask_count`` starts the emit
    block". Pinned here so the secondary signal can't silently
    regress.
    """
    from decoding_sandbox.core.engine import GenStep
    from decoding_sandbox.core.types import StepResult

    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={},
        has_completions=True,
        supports_new_logprobs=True,
        supports_combined_echo_stream=True,
    )
    chunks = [
        # Echo entries: NO text_offset, NO sampling_logprob/mask.
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "X",
                                "token_id": 1,
                                "logprob": 0.0,
                                "top_logprobs": [],
                            }
                        ]
                    }
                }
            ]
        },
        # Emit entry: sampling_mask_count present.
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "Y",
                                "token_id": 2,
                                "logprob": -0.1,
                                "sampling_mask_count": 50,
                                "top_logprobs": [],
                            }
                        ]
                    },
                    "finish_reason": "length",
                }
            ]
        },
    ]
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines(chunks))])
    # Use a prompt whose char length doesn't help split (single ascii
    # token would make the cumulative-text heuristic fire too early or
    # too late depending on prompt). A 5-char prompt + 1-char echo
    # token leaves running_text_len=1<5 after the echo, so the
    # cumulative-text path would NOT split there -- only the
    # sampling-signal path puts the boundary in the right place.
    out = list(
        backend.stream_native_with_echo(
            "12345",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
        )
    )
    step_results = [x for x in out if isinstance(x, StepResult)]
    gen_steps = [x for x in out if isinstance(x, GenStep)]
    assert [s.chosen.token_id for s in step_results] == [1]
    assert [g.decision.token_id for g in gen_steps] == [2]


def test_stream_native_with_echo_forwards_echo_last(monkeypatch) -> None:
    """``echo_last`` reaches the wire when the provider opts in."""
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={},
        has_completions=True,
        supports_combined_echo_stream=True,
    )
    # Manually toggle echo_last support on the provider (the
    # always-on Fireworks defaults already set it; for the test
    # helper's bare-bones ProviderConfig we flip it here).
    backend.provider.supports_echo_last = True
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native_with_echo(
            "hello",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            echo_last=4,
        )
    )
    body = mock.calls[-1]["kwargs"]["json"]
    assert body["echo_last"] == 4


def test_stream_native_forwards_service_tier_when_supported(monkeypatch) -> None:
    """``service_tier`` flows to the body only when the provider supports it."""
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, supports_service_tier=True
    )
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            service_tier="priority",
        )
    )
    assert mock.calls[-1]["kwargs"]["json"]["service_tier"] == "priority"


def test_stream_native_drops_service_tier_when_unsupported(monkeypatch) -> None:
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            service_tier="priority",
        )
    )
    assert "service_tier" not in mock.calls[-1]["kwargs"]["json"]


def test_stream_native_forwards_prompt_cache_key_when_supported(monkeypatch) -> None:
    """``prompt_cache_key`` keeps requests on the same KV-cache-warm replica."""
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, supports_prompt_cache_key=True
    )
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            prompt_cache_key="manual-session-abc",
        )
    )
    body = mock.calls[-1]["kwargs"]["json"]
    assert body["prompt_cache_key"] == "manual-session-abc"


def test_stream_native_session_affinity_headers_set_on_supported_provider(monkeypatch) -> None:
    """A ``session_id`` becomes ``x-session-affinity`` + R3 multi-turn headers.

    Two headers, one body key: ``x-session-affinity`` pins the request
    to a specific replica (sticky routing) and
    ``x-multi-turn-session-id`` triggers MoE Router Replay so the
    expert-routing trace is reused across turns. Both are
    Fireworks-extensions; providers without ``supports_session_affinity``
    must NOT see either header (some upstreams reject unknown headers).
    """
    backend, mock = _make_oc_backend(
        monkeypatch, routes={}, has_completions=True, supports_session_affinity=True
    )
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            session_id="sess-42",
        )
    )
    headers = mock.calls[-1]["kwargs"].get("headers") or {}
    assert headers.get("x-session-affinity") == "sess-42"
    assert headers.get("x-multi-turn-session-id") == "sess-42"


def test_stream_native_session_affinity_headers_omitted_when_unsupported(monkeypatch) -> None:
    backend, mock = _make_oc_backend(monkeypatch, routes={}, has_completions=True)
    _attach_stream_factory(mock, [_MockStreamResponse(200, _sse_lines([]))])
    list(
        backend.stream_native(
            "p",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=1,
            top_k=1,
            session_id="sess-42",
        )
    )
    headers = mock.calls[-1]["kwargs"].get("headers") or {}
    assert "x-session-affinity" not in headers
    assert "x-multi-turn-session-id" not in headers


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
    # Critical: the streaming path opens one HTTP connection -- the
    # ``requests`` counter MUST tick. Earlier this was zero because
    # ``_iter_completions_stream`` bypassed ``_request`` and never
    # called ``record_request`` itself, so the UI happily reported
    # "0 requests, 20 tokens" for a perfectly working native stream.
    assert sink["requests"] == 1


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
