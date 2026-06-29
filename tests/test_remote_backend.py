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

import httpx
import pytest
from fastapi.testclient import TestClient

from decoding_sandbox.backends.remote import (
    RemoteBackend,
    RemoteBackendError,
    RemoteStreamTimeoutError,
)
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

        def next_distribution(self, token_ids, top_k, *, watch_ids=()):
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
    [gs] = list(remote.stream_generate("ab", "greedy", {}, max_tokens=1))
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
        def next_distribution(self, token_ids, top_k, *, watch_ids=()):
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


def test_stream_generate_forwards_watch_ids_and_prefix_token_ids_in_body() -> None:
    """The unified Decode workbench's manual mode rides on
    ``prefix_token_ids``; the inspect/generate watch panel rides on
    ``watch_ids``. Both have to make it into the JSON body posted to
    ``/v1/generate/stream`` -- if they don't, dsbx-serve has nothing to
    forward to :func:`core.engine.generate` and the watched columns
    silently render as "—" no matter what the backend supports.

    We use a stub httpx-style client that captures the JSON body, then
    drive a single token through so the call actually completes. The
    response shape mirrors a real dsbx-serve stream.
    """
    captured: dict[str, object] = {}

    def lines():
        # One step + done frame -- enough to make stream_generate yield.
        yield 'data: {"event":"step","step":{"step":0,"tokens_before":[97,98],"step_result":{"position":2,"candidates":[{"token_id":99,"text":"X","logprob":-0.1,"rank":0,"is_special":false,"sampling_mask_count":null}],"is_full_vocab":false,"chosen":{"token_id":99,"text":"X","logprob":-0.1,"rank":0,"is_special":false,"sampling_mask_count":null},"context_text":"","watched":[]},"decision":{"token_id":99,"token_text":"X","kept":[],"greedy_token_id":99,"note":""},"stop_reason":"max_tokens"}}'
        yield 'data: {"event":"done","stop_reason":"max_tokens"}'

    fake = _StreamClient(info_payload=_info_payload(), line_factory=lines)
    # Intercept the JSON body the stream POST receives.
    original_stream = fake.stream

    def capturing_stream(method, path, *, json=None, **kw):
        captured["json"] = json
        return original_stream(method, path, json=json, **kw)

    fake.stream = capturing_stream  # type: ignore[assignment]
    rb = RemoteBackend("http://test", client=fake)
    list(
        rb.stream_generate(
            "ab",
            "greedy",
            {},
            max_tokens=1,
            watch_ids=[111, 222],
            prefix_token_ids=[42, 43, 44],
        )
    )
    body = captured.get("json")
    assert isinstance(body, dict)
    assert body.get("watch_ids") == [111, 222]
    assert body.get("prefix_token_ids") == [42, 43, 44]


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


# --------------------------------------------------------------------------- #
# Streaming: timeout + cancellation cleanup
# --------------------------------------------------------------------------- #
class _StreamClient:
    """Minimal httpx.Client stand-in that supports the surface our
    streaming code uses: ``.get`` for /v1/info, ``.stream`` returning a
    context-managed response whose ``iter_lines`` is fed by a caller-
    supplied callable.

    The point is to drive ``RemoteBackend.stream_generate`` through its
    happy path AND its ReadTimeout / GeneratorExit paths without
    standing up a server, so we can assert (a) the right timeout knob
    reaches httpx, (b) ``RemoteStreamTimeoutError`` is raised on a hang, and
    (c) closing the generator closes the underlying connection.
    """

    def __init__(self, *, info_payload: dict, line_factory) -> None:
        self._info = info_payload
        self._line_factory = line_factory
        self.last_timeout: object | None = None
        self.closed_streams = 0

    def get(self, path: str, **_: object) -> _StreamResp:
        return _StreamResp(200, payload=self._info, owner=self)

    def post(self, path: str, *, json=None, **_: object) -> _StreamResp:
        # Not actually used in these tests; the streaming path uses
        # ``self.stream``. Defined anyway because RemoteBackend's
        # constructor relies on ``post`` for the ``_get_info`` call's
        # error fallback path -- not exercised here but cheap to leave.
        return _StreamResp(200, payload={}, owner=self)

    def stream(
        self,
        method: str,
        path: str,
        *,
        json=None,
        timeout=None,
        **_: object,
    ) -> _StreamCtx:
        self.last_timeout = timeout
        return _StreamCtx(self._line_factory(), owner=self)

    def close(self) -> None:
        return None


class _StreamResp:
    def __init__(self, status_code: int, *, payload, owner: _StreamClient) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}
        self._owner = owner

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 300):
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=None,
                response=None,  # type: ignore[arg-type]
            )


class _StreamCtx:
    def __init__(self, line_iter, *, owner: _StreamClient) -> None:
        self._lines = line_iter
        self.status_code = 200
        self._owner = owner

    def __enter__(self) -> _StreamCtx:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._owner.closed_streams += 1
        return False

    def iter_lines(self):
        # Yield from the caller-provided iterator. Tests inject one of:
        #   - a generator yielding SSE lines + then returning (happy path)
        #   - a function raising httpx.ReadTimeout (hung upstream)
        #   - a generator yielding one line then pausing forever
        yield from self._lines

    def read(self) -> bytes:
        return b""


def _info_payload() -> dict:
    """Minimal /v1/info shape ``RemoteBackend`` needs to construct."""
    return {
        "backend_kind": "test",
        "engine_version": "0",
        "loaded_model": None,
        "capabilities": {
            "name": "test",
            "full_vocab": True,
            "prompt_logprobs": True,
            "max_top_logprobs": 5,
            "can_force_token": False,
            "notes": "",
            "eos_token_ids": [],
        },
    }


def test_stream_generate_uses_short_per_frame_read_timeout() -> None:
    """The streaming POST must carry an explicit ``httpx.Timeout`` whose
    ``read`` knob equals the backend's per-frame timeout, NOT the
    client's wider default. This is what lets the web layer free its
    sync generator within seconds of an upstream hang instead of the
    full 120 s client default."""

    def lines():
        # Empty stream + immediate done so the iterator returns cleanly.
        yield 'data: {"event": "done"}'
        yield ""

    fake = _StreamClient(info_payload=_info_payload(), line_factory=lines)
    rb = RemoteBackend("http://test", client=fake, stream_read_timeout=12.5)
    list(rb.stream_generate("hi", "greedy", {}, max_tokens=1))
    timeout = fake.last_timeout
    assert isinstance(timeout, httpx.Timeout)
    # httpx stores values per-phase; the read knob is what matters here.
    assert timeout.read == pytest.approx(12.5)
    # Connect kept short on purpose so a black-hole gateway surfaces fast.
    assert (timeout.connect or 0) <= 15.0


def test_stream_generate_translates_read_timeout_to_remote_stream_timeout() -> None:
    """A hung upstream surfaces as ``RemoteStreamTimeoutError`` (subclass of
    RemoteBackendError) with a message naming the configured budget --
    matches the user-facing error the SSE done frame now carries."""

    def lines():
        # Mimic httpx raising ReadTimeout from inside iter_lines when no
        # bytes arrive within the configured read budget. (httpx itself
        # surfaces this as ReadTimeout from the underlying transport.)
        raise httpx.ReadTimeout("read timed out")
        yield  # pragma: no cover -- unreachable, but marks this a generator

    fake = _StreamClient(info_payload=_info_payload(), line_factory=lines)
    rb = RemoteBackend("http://test", client=fake, stream_read_timeout=7.0)
    with pytest.raises(RemoteStreamTimeoutError, match=">7s"):
        list(rb.stream_generate("hi", "greedy", {}, max_tokens=1))
    # The ``with self._client.stream(...)`` context manager MUST close
    # even on the timeout path -- otherwise the leaked httpx Response
    # would still pin a connection to the dead upstream.
    assert fake.closed_streams == 1


def test_stream_generate_closes_upstream_on_generator_close() -> None:
    """Cancelling the iterator mid-stream closes the underlying httpx
    response. That's the chain the web layer relies on so the
    browser's "stop" click actually RSTs the wire to the upstream
    ``dsbx serve`` (instead of letting the connection limp on until
    the response naturally completes)."""

    def lines():
        # Yield one usable line, then "park" forever waiting for more
        # data. The test will close the generator after consuming the
        # first frame; we should never reach the second yield.
        yield (
            'data: {"event": "step", "step": {"step": 0, "tokens_before": [], '
            '"step_result": {"position": 0, "candidates": [], "is_full_vocab": false}, '
            '"decision": {"token_id": 1, "token_text": "x", "kept": []}, '
            '"stop_reason": null}}'
        )
        yield ""
        # If the generator wasn't closed, we'd block forever here. The
        # test asserting ``closed_streams == 1`` proves we did close.
        while True:  # pragma: no cover -- unreachable when close works
            yield ""

    fake = _StreamClient(info_payload=_info_payload(), line_factory=lines)
    rb = RemoteBackend("http://test", client=fake)
    gen = rb.stream_generate("hi", "greedy", {}, max_tokens=99)
    first = next(gen)
    assert first.step == 0
    gen.close()  # mirrors what Starlette does on client disconnect
    assert fake.closed_streams == 1


def test_stream_read_timeout_defaults_to_class_constant() -> None:
    """A backend constructed without an explicit ``stream_read_timeout``
    falls back to the class default. Keeps the public surface stable
    for callers that don't care about the knob."""
    rb = RemoteBackend(
        "http://test",
        client=_StreamClient(info_payload=_info_payload(), line_factory=lambda: iter([])),
    )
    assert rb._stream_read_timeout == RemoteBackend.DEFAULT_STREAM_READ_TIMEOUT
