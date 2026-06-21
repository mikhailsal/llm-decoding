"""Tests for the HTTP-talking backends (OpenAI-compat and llama.cpp).

We patch in a ``MockHTTPClient`` (see ``tests/fakes.py``) so we exercise the
real JSON-parsing logic without standing up servers. Every test asserts both
(a) the request shape we send and (b) the StepResult we build from the canned
response.
"""

from __future__ import annotations

import math

import pytest

from decoding_sandbox.backends import llamacpp as llamacpp_mod
from decoding_sandbox.backends import openai_compat as oc_mod
from decoding_sandbox.backends.llamacpp import LlamaCppBackend
from decoding_sandbox.backends.openai_compat import OpenAICompatBackend
from decoding_sandbox.core.config import ProviderConfig
from tests.fakes import MockHTTPClient


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
) -> tuple[OpenAICompatBackend, MockHTTPClient]:
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
    backend = OpenAICompatBackend(prov, model="test/m")
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
    backend, mock = _make_oc_backend(
        monkeypatch,
        routes={
            ("POST", "/completions"): {
                "choices": [
                    {
                        "logprobs": {
                            "tokens": ["The", " cap", "ital"],
                            "token_logprobs": [None, -1.0, -2.0],
                            "top_logprobs": [
                                None,
                                {" cap": -1.0, " other": -3.0},
                                {"ital": -2.0, "stuff": -4.0},
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

    assert len(steps) == 2
    s1, s2 = steps
    assert s1.context_text == "The"
    assert s1.chosen is not None
    assert s1.chosen.text == " cap"
    assert s1.chosen.logprob == pytest.approx(-1.0)
    assert s2.chosen.text == "ital"
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
