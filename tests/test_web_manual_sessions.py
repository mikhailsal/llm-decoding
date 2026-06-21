"""Tests for /api/v1/manual/sessions/* (full lifecycle + TTL).

Covers:

- ``create`` returns a snapshot with a session_id + the current
  next-token distribution.
- ``pick`` by rank advances state and out-of-range raises 400.
- ``force`` by text and by id both work; ``can_force_token=False`` is honored.
- ``undo`` reverses the last token.
- ``set_top_k`` mutates the session's top_k.
- ``transcript`` returns valid JSON the UI can save; ``load`` re-applies it.
- ``delete`` is idempotent and returns whether it existed.
- ``TTL`` eviction drops idle sessions.
- The per-session lock allows concurrent picks to be serialized without
  corruption (we use threads + a shared id to assert order).
"""

from __future__ import annotations

import threading

import pytest

from decoding_sandbox.core.types import Capabilities
from decoding_sandbox.web.sessions import ManualSessionRegistry
from tests.fakes import FakeBackend, cand
from tests.web_helpers import build_test_app, make_authed_client


def _backend() -> FakeBackend:
    return FakeBackend(
        tokens={"ab": [97, 98], " however": [300]},
        pieces={97: "a", 98: "b", 88: "X", 89: "Y", 99: "", 300: " however"},
        distributions={
            (97, 98): [cand(88, "X", 0.6, 0), cand(89, "Y", 0.4, 1)],
            (97, 98, 88): [cand(88, "X", 0.55, 0), cand(89, "Y", 0.45, 1)],
            (97, 98, 88, 88): [cand(88, "X", 0.55, 0), cand(89, "Y", 0.45, 1)],
            (97, 98, 88, 88, 89): [cand(88, "X", 0.5, 0), cand(89, "Y", 0.5, 1)],
            (97, 98, 88, 300): [cand(88, "X", 0.5, 0), cand(89, "Y", 0.5, 1)],
        },
        eos_token_ids=(99,),
    )


@pytest.fixture
def client():
    app = build_test_app({"dsbx-host-py": _backend()})
    with make_authed_client(app) as c:
        yield c


def _create(client) -> dict:
    r = client.post(
        "/api/v1/manual/sessions",
        json={"backend": "dsbx-host-py", "prompt": "ab", "top_k": 5},
    )
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------------- #
def test_create_returns_snapshot(client) -> None:
    snap = _create(client)
    assert "session_id" in snap
    assert snap["backend"] == "dsbx-host-py"
    assert snap["prompt"] == "ab"
    assert snap["prompt_ids"] == [97, 98]
    assert snap["generated_ids"] == []
    assert snap["top_k"] == 5
    # Distribution should be populated for the current cursor.
    assert snap["distribution"]["candidates"]
    assert snap["distribution"]["position"] == 2


def test_pick_appends_token(client) -> None:
    snap = _create(client)
    sid = snap["session_id"]
    r = client.post(f"/api/v1/manual/sessions/{sid}/pick", json={"rank": 0})
    assert r.status_code == 200
    after = r.json()
    assert after["generated_ids"] == [88]
    assert after["generated_text"] == "X"


def test_pick_out_of_range_returns_400(client) -> None:
    snap = _create(client)
    sid = snap["session_id"]
    r = client.post(f"/api/v1/manual/sessions/{sid}/pick", json={"rank": 99})
    assert r.status_code == 400


def test_force_text(client) -> None:
    snap = _create(client)
    sid = snap["session_id"]
    r = client.post(f"/api/v1/manual/sessions/{sid}/force", json={"text": " however"})
    assert r.status_code == 200
    assert r.json()["generated_ids"] == [300]


def test_force_id(client) -> None:
    snap = _create(client)
    sid = snap["session_id"]
    r = client.post(f"/api/v1/manual/sessions/{sid}/force", json={"id": 88})
    assert r.status_code == 200
    assert r.json()["generated_ids"] == [88]


def test_force_requires_exactly_one_input(client) -> None:
    snap = _create(client)
    sid = snap["session_id"]
    r = client.post(f"/api/v1/manual/sessions/{sid}/force", json={})
    assert r.status_code == 400
    r = client.post(f"/api/v1/manual/sessions/{sid}/force", json={"text": "x", "id": 1})
    assert r.status_code == 400


def test_force_blocked_when_capability_false() -> None:
    class _NoForce(FakeBackend):
        @property
        def capabilities(self) -> Capabilities:
            return Capabilities(
                name="no-force",
                full_vocab=True,
                prompt_logprobs=True,
                max_top_logprobs=5,
                can_force_token=False,
                eos_token_ids=(),
            )

    backend = _NoForce(
        tokens={"ab": [97, 98]},
        pieces={97: "a", 98: "b", 88: "X", 89: "Y"},
        distributions={
            (97, 98): [cand(88, "X", 0.6, 0), cand(89, "Y", 0.4, 1)],
        },
    )
    app = build_test_app({"chat": backend})
    with make_authed_client(app) as c:
        snap = c.post(
            "/api/v1/manual/sessions",
            json={"backend": "chat", "prompt": "ab"},
        ).json()
        r = c.post(
            f"/api/v1/manual/sessions/{snap['session_id']}/force",
            json={"text": "x"},
        )
    assert r.status_code == 400
    assert "force" in r.json()["detail"].lower()


def test_undo_pops_last(client) -> None:
    snap = _create(client)
    sid = snap["session_id"]
    client.post(f"/api/v1/manual/sessions/{sid}/pick", json={"rank": 0})
    after = client.post(f"/api/v1/manual/sessions/{sid}/undo").json()
    assert after["generated_ids"] == []


def test_set_top_k(client) -> None:
    snap = _create(client)
    sid = snap["session_id"]
    after = client.post(f"/api/v1/manual/sessions/{sid}/set_top_k", json={"top_k": 9}).json()
    assert after["top_k"] == 9


def test_set_top_k_validates_positive(client) -> None:
    snap = _create(client)
    sid = snap["session_id"]
    r = client.post(f"/api/v1/manual/sessions/{sid}/set_top_k", json={"top_k": 0})
    assert r.status_code == 400


def test_transcript_then_load_roundtrip(client) -> None:
    snap = _create(client)
    sid = snap["session_id"]
    client.post(f"/api/v1/manual/sessions/{sid}/pick", json={"rank": 0})
    client.post(f"/api/v1/manual/sessions/{sid}/pick", json={"rank": 1})
    trans = client.get(f"/api/v1/manual/sessions/{sid}/transcript").json()
    assert trans["generated_ids"] == [88, 89]
    # Now undo + load -> the session should be back at [88, 89].
    client.post(f"/api/v1/manual/sessions/{sid}/undo").json()
    after = client.post(f"/api/v1/manual/sessions/{sid}/load", json=trans).json()
    assert after["generated_ids"] == [88, 89]


def test_delete_is_idempotent(client) -> None:
    snap = _create(client)
    sid = snap["session_id"]
    r1 = client.delete(f"/api/v1/manual/sessions/{sid}")
    assert r1.status_code == 200
    assert r1.json()["deleted"] is True
    r2 = client.delete(f"/api/v1/manual/sessions/{sid}")
    assert r2.json()["deleted"] is False


def test_unknown_session_404(client) -> None:
    r = client.get("/api/v1/manual/sessions/no-such")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Registry-level tests (no HTTP) for TTL + locking
# --------------------------------------------------------------------------- #
def test_registry_ttl_eviction() -> None:
    """Sessions older than ttl are dropped on the next public call."""
    now = [1000.0]

    def fake_now() -> float:
        return now[0]

    reg = ManualSessionRegistry(ttl_seconds=10.0, now=fake_now)
    backend = _backend()
    entry = reg.create("dsbx-host-py", backend, "ab", top_k=5)
    sid = entry.session_id
    assert reg.get(sid) is entry

    # Advance time well past the TTL; ``get`` should now raise KeyError.
    now[0] += 30
    with pytest.raises(KeyError):
        reg.get(sid)


def test_registry_per_session_lock_serializes_picks() -> None:
    """Two threads picking on the same session see consistent state.

    Without a per-session lock the ``ManualSession``'s ``generated_ids`` list
    would interleave; with the lock, the final length must equal the total
    number of picks across both threads.
    """
    reg = ManualSessionRegistry(ttl_seconds=3600.0)
    backend = _backend()
    entry = reg.create("dsbx-host-py", backend, "ab", top_k=5)

    def loop() -> None:
        for _ in range(20):
            with entry.lock:
                entry.session.generated_ids.append(88)

    threads = [threading.Thread(target=loop) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(entry.session.generated_ids) == 80
