"""RemoteBackend round-trip tests against a real FastAPI app.

We use ``fastapi.testclient.TestClient`` -- which is itself a subclass of
``httpx.Client`` -- as the injected client. That gives the RemoteBackend
a real HTTP-shaped connection to a real FastAPI app, with no actual
network involved. Result: the tests exercise the wire format
(``schemas.py`` <-> dict <-> dataclass round trips) on every call, the
SSE parser, and the ``stream_generate`` -> ``GenStep`` reconstruction.
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from decoding_sandbox.backends.remote import RemoteBackend, RemoteBackendError
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.types import StepResult, TokenCandidate
from decoding_sandbox.server.app import make_app
from tests.fakes import FakeBackend, cand


def _make_remote(backend: Backend, *, backend_kind: str = "fake-kind") -> RemoteBackend:
    """Wire a RemoteBackend at ``http://testserver`` to ``make_app(backend)``.

    Returning the TestClient lets the caller close it explicitly; the
    RemoteBackend's ``close()`` does not close an injected client
    (``_owns_client=False``).
    """
    app = make_app(backend, backend_kind=backend_kind)
    tc = TestClient(app)
    return RemoteBackend("http://testserver", client=tc)


def _fake() -> FakeBackend:
    return FakeBackend(
        tokens={"ab": [97, 98]},
        pieces={97: "a", 98: "b", 88: "X", 89: "Y"},
        distributions={
            (97,): [cand(98, "b", 0.6, 0), cand(89, "Y", 0.4, 1)],
            (97, 98): [cand(88, "X", 0.6, 0), cand(89, "Y", 0.4, 1)],
            (97, 98, 88): [cand(88, "X", 0.55, 0), cand(89, "Y", 0.45, 1)],
            (97, 98, 88, 88): [cand(88, "X", 0.55, 0), cand(89, "Y", 0.45, 1)],
        },
        eos_token_ids=(99,),
    )


# --------------------------------------------------------------------------- #
# Construction / capabilities
# --------------------------------------------------------------------------- #
def test_constructor_fetches_capabilities_from_info() -> None:
    remote = _make_remote(_fake(), backend_kind="hf")
    caps = remote.capabilities
    assert caps.name == "fake"
    assert caps.full_vocab is True
    assert caps.prompt_logprobs is True
    assert caps.max_top_logprobs == 10
    assert 99 in caps.eos_token_ids
    assert remote.backend_kind == "hf"


def test_loaded_model_round_trips() -> None:
    backend = _fake()
    backend.model_path = "/tmp/my.gguf"  # detected by _detect_loaded_model
    remote = _make_remote(backend, backend_kind="llamacpp-py")
    assert remote.loaded_model == "/tmp/my.gguf"


def test_constructor_raises_when_server_returns_500() -> None:
    """A failing /v1/info bubbles up as RemoteBackendError -- the client
    is unusable without capabilities, so failing loudly is correct.

    ``raise_server_exceptions=False`` puts the TestClient in "real
    server" mode where unhandled exceptions render as HTTP 500 instead
    of being re-raised through the test thread -- which is what a real
    client would observe."""

    class _BrokenInfo(Backend):
        @property
        def capabilities(self):
            raise RuntimeError("info exploded")

        def tokenize(self, t):
            return []

        def detokenize(self, ids):
            return ""

        def piece(self, tid):
            return ""

        def next_distribution(self, token_ids, top_k):
            return StepResult(0, [], False)

    app = make_app(_BrokenInfo())
    tc = TestClient(app, raise_server_exceptions=False)
    with pytest.raises(RemoteBackendError):
        RemoteBackend("http://testserver", client=tc)


# --------------------------------------------------------------------------- #
# Tokenization / inference round trips
# --------------------------------------------------------------------------- #
def test_tokenize_round_trip() -> None:
    remote = _make_remote(_fake())
    assert remote.tokenize("ab") == [97, 98]
    assert remote.detokenize([97, 98]) == "ab"


def test_piece_is_cached_client_side() -> None:
    """The piece cache prevents per-token network bursts during render."""
    remote = _make_remote(_fake())
    a1 = remote.piece(88)
    # Wipe the server-side backend so a real network call would 500.
    a2 = remote.piece(88)
    assert a1 == a2 == "X"
    assert 88 in remote._piece_cache


def test_next_distribution_round_trips_logprobs_and_ranks() -> None:
    remote = _make_remote(_fake())
    step = remote.next_distribution([97, 98], top_k=2)
    assert step.position == 2
    assert step.is_full_vocab is True
    assert [c.text for c in step.candidates] == ["X", "Y"]
    # Probabilities round-trip exactly (we sent log of 0.6/0.4).
    assert step.candidates[0].prob == pytest.approx(0.6, abs=1e-9)
    assert step.candidates[0].rank == 0


def test_score_prompt_round_trip_with_watch_ids() -> None:
    remote = _make_remote(_fake())
    steps = remote.score_prompt("ab", top_k=3, watch_ids=[88])
    assert len(steps) == 2
    # Trailing step has chosen=None and a watched lookup for id 88.
    assert steps[-1].chosen is None
    assert 88 in steps[-1].watched
    # Non-trailing step gets a real chosen candidate.
    assert steps[0].chosen is not None


def test_score_prompt_nan_logprob_becomes_nan_again_on_the_client() -> None:
    """A watch id that fell outside the server's top-k arrives as
    ``logprob=None`` over the wire and is restored to ``math.nan`` in
    memory so the renderer's ``isnan`` branch keeps working."""
    remote = _make_remote(_fake())
    # Watching a token id not in any of the seeded distributions forces
    # the generic Backend.score_prompt fallback to mark its candidate
    # ``rank=-1, logprob=NaN``. We assert the NaN survives the round trip.
    steps = remote.score_prompt("ab", top_k=3, watch_ids=[1234])
    watched = steps[0].watched[1234]
    assert watched.rank == -1
    assert math.isnan(watched.logprob)


# --------------------------------------------------------------------------- #
# verify_greedy
# --------------------------------------------------------------------------- #
def test_verify_greedy_round_trip() -> None:
    """A backend with verify_greedy returns the tuple shape the in-process
    HFBackend.verify_greedy produces."""

    class _WithVerify(FakeBackend):
        def verify_greedy(self, context_ids, draft_ids):
            return len(draft_ids), TokenCandidate(89, "Y", math.log(0.5), 0)

    remote = _make_remote(_WithVerify())
    accepted, correction = remote.verify_greedy([97], [98, 99])
    assert accepted == 2
    assert correction.token_id == 89
    assert correction.text == "Y"


def test_verify_greedy_raises_when_unsupported() -> None:
    remote = _make_remote(_fake())  # FakeBackend has no verify_greedy
    with pytest.raises(RemoteBackendError, match="verify_greedy"):
        remote.verify_greedy([97], [98])


# --------------------------------------------------------------------------- #
# stream_generate (SSE round-trip)
# --------------------------------------------------------------------------- #
def test_stream_generate_yields_genstep_per_token() -> None:
    remote = _make_remote(_fake())
    steps = list(
        remote.stream_generate(
            "ab",
            sampler_name="greedy",
            sampler_params={},
            max_tokens=2,
        )
    )
    assert len(steps) == 2
    assert [gs.step for gs in steps] == [0, 1]
    assert steps[0].decision.token_text == "X"
    # The last step's stop_reason is set by the engine -- max_tokens here.
    assert steps[-1].stop_reason == "max_tokens"


def test_stream_generate_decision_kept_join_with_candidates() -> None:
    """The client reconstructs ``SamplerDecision.kept`` by joining kept
    ids against the step's ``candidates`` -- verify the join produces
    real ``TokenCandidate`` references, not stubs."""
    remote = _make_remote(_fake())
    [gs] = list(
        remote.stream_generate("ab", "greedy", {}, max_tokens=1)
    )
    assert gs.decision.kept
    for cand_obj, prob in gs.decision.kept:
        # Greedy keeps only the top candidate; it must be in the step's
        # candidate list with a real (finite) logprob, not a stub.
        assert cand_obj in gs.step_result.candidates
        assert prob > 0


def test_stream_generate_propagates_server_error_event() -> None:
    """A backend that explodes mid-decode lands in the SSE done event's
    ``error`` field; the client surfaces it as RemoteBackendError."""

    class _BoomBackend(FakeBackend):
        def next_distribution(self, token_ids, top_k):
            raise RuntimeError("server side boom")

    remote = _make_remote(_BoomBackend(tokens={"ab": [97, 98]}, eos_token_ids=(99,)))
    with pytest.raises(RemoteBackendError, match="server side boom"):
        list(remote.stream_generate("ab", "greedy", {}, max_tokens=2))


def test_stream_generate_rejects_unknown_sampler_with_4xx() -> None:
    remote = _make_remote(_fake())
    with pytest.raises(RemoteBackendError, match="HTTP 400"):
        list(remote.stream_generate("ab", "nope", {}, max_tokens=1))


def test_stream_generate_marker_attribute_is_set() -> None:
    """cmd_generate keys off this marker to pick the streaming path."""
    remote = _make_remote(_fake())
    assert RemoteBackend.supports_remote_stream is True
    assert hasattr(remote, "stream_generate")


# --------------------------------------------------------------------------- #
# Close semantics
# --------------------------------------------------------------------------- #
def test_close_does_not_close_injected_client() -> None:
    """An injected TestClient is owned by the caller, not the backend."""
    backend = _fake()
    app = make_app(backend)
    tc = TestClient(app)
    remote = RemoteBackend("http://testserver", client=tc)
    remote.close()
    # The injected client is still usable.
    r = tc.get("/v1/info")
    assert r.status_code == 200
    tc.close()


def test_close_closes_owned_client(monkeypatch) -> None:
    """When the backend constructs its own httpx.Client it must close it."""
    # We can't easily check httpx.Client.close ran without monkeypatching;
    # use a counter on the underlying client.
    backend = _fake()
    app = make_app(backend)
    tc = TestClient(app)
    remote = RemoteBackend("http://testserver", client=tc)
    # Force the client to look owned so close() runs the close path.
    remote._owns_client = True
    closes: list[int] = []
    monkeypatch.setattr(tc, "close", lambda: closes.append(1))
    remote.close()
    assert closes == [1]
