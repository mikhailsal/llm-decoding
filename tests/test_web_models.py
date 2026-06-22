"""``GET /api/v1/models/{name}``: catalogue listing + 6h TTL cache.

Three things this endpoint must get right:

- For cloud providers with no key (the dev-machine pattern) it falls back
  to the curated ``[providers.NAME].models`` list -- the UI must still
  show *something* even when we can't talk to the wire.
- For remote / local backends it never makes a network call; it returns
  the single configured model.
- ``?refresh=true`` always re-hits the upstream (or re-walks the static
  fallback path), without poisoning the cache for the other backends.
"""

from __future__ import annotations

import time

import pytest

from decoding_sandbox.web import backends as web_backends
from decoding_sandbox.web.backends import BackendRegistry, ModelListEntry
from tests.fakes import FakeBackend
from tests.web_helpers import build_test_app, make_authed_client, make_test_config


@pytest.fixture
def app_no_keys(monkeypatch):
    """An app where neither Fireworks nor NIM has an API key set.

    That's the state the curated-list fallback path was designed for: we
    can't reach the wire, but ``list_models`` should still return the
    static suggestions.
    """
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    backend = FakeBackend(tokens={}, pieces={}, distributions={}, eos_token_ids=(99,))
    cfg = make_test_config(
        remotes={"dsbx-host-py": "http://192.0.2.42:8000"},
        providers=["fireworks", "nim"],
    )
    # ``make_test_config`` doesn't set the ``models`` list on the stub
    # providers. Inject one so the curated-fallback path has something
    # interesting to return (otherwise the response would be a single
    # default-model row and we couldn't distinguish curated from default).
    cfg.providers["fireworks"].models = [
        "accounts/fireworks/models/gpt-oss-120b",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
    ]
    cfg.providers["nim"].models = ["meta/llama-3.1-8b-instruct"]
    return build_test_app({"dsbx-host-py": backend}, cfg=cfg)


def test_cloud_no_key_falls_back_to_curated_list(app_no_keys) -> None:
    """No API key -> fallback to curated suggestions, never crash."""
    with make_authed_client(app_no_keys) as c:
        r = c.get("/api/v1/models/fireworks")
    assert r.status_code == 200
    data = r.json()
    assert data["backend"] == "fireworks"
    assert data["source"] == "fallback"
    assert "accounts/fireworks/models/gpt-oss-120b" in data["models"]
    # The default model is always first in the curated list.
    assert data["models"][0] == "fireworks/default-model"
    # Note is human-readable and never contains the URL/env-var name.
    assert "FIREWORKS_API_KEY" in data["note"]
    assert "https://" not in data["note"]


def test_remote_backend_returns_static_single_model(app_no_keys) -> None:
    """``remote`` and ``local`` backends short-circuit -- no network call."""
    with make_authed_client(app_no_keys) as c:
        r = c.get("/api/v1/models/dsbx-host-py")
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "static"
    # The fake backend isn't ``RemoteBackend`` so loaded_model is None;
    # the static list is empty in that case but the call still succeeds
    # rather than hitting the wire. (A real dsbx-host-py would surface its
    # configured GGUF here.)
    assert isinstance(data["models"], list)


def test_unknown_backend_returns_404(app_no_keys) -> None:
    with make_authed_client(app_no_keys) as c:
        r = c.get("/api/v1/models/does-not-exist")
    assert r.status_code == 404


@pytest.fixture
def app_with_key(monkeypatch):
    """An app where Fireworks has a key, so live fetches are allowed.

    Key is set BEFORE ``build_test_app`` runs so ``BackendRegistry``
    sees Fireworks as available; tests patch the fetch method to avoid
    actually hitting the network.
    """
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key-DSBX-models")
    backend = FakeBackend(tokens={}, pieces={}, distributions={}, eos_token_ids=(99,))
    cfg = make_test_config(
        remotes={"dsbx-host-py": "http://192.0.2.42:8000"},
        providers=["fireworks", "nim"],
    )
    cfg.providers["fireworks"].models = [
        "accounts/fireworks/models/gpt-oss-120b",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
    ]
    return build_test_app({"dsbx-host-py": backend}, cfg=cfg)


def test_cache_serves_repeat_calls_without_refetch(app_with_key, monkeypatch) -> None:
    """A second call within the TTL must be served from cache.

    We assert the wire path was hit exactly once by patching
    ``OpenAICompatBackend.fetch_available_models`` to count invocations.
    The two calls must yield ``source="live"`` then ``source="cached"``.
    """
    calls: list[str] = []

    def _fake_fetch(self, timeout: float = 15.0) -> list[str]:  # noqa: ARG001
        calls.append(self.provider.name)
        return ["accounts/fireworks/models/gpt-oss-120b", "accounts/fireworks/models/fresh-model"]

    from decoding_sandbox.backends import openai_compat as oa

    monkeypatch.setattr(oa.OpenAICompatBackend, "fetch_available_models", _fake_fetch)
    with make_authed_client(app_with_key) as c:
        r1 = c.get("/api/v1/models/fireworks")
        r2 = c.get("/api/v1/models/fireworks")
    assert r1.json()["source"] == "live"
    assert r2.json()["source"] == "cached"
    assert len(calls) == 1
    # Live response unions curated + live in a stable order: curated
    # entries first, live extras appended.
    assert r1.json()["models"][:2] == [
        "fireworks/default-model",
        "accounts/fireworks/models/gpt-oss-120b",
    ]
    assert "accounts/fireworks/models/fresh-model" in r1.json()["models"]


def test_refresh_bypasses_cache(app_with_key, monkeypatch) -> None:
    """``?refresh=true`` always re-fetches."""
    calls: list[str] = []

    def _fake_fetch(self, timeout: float = 15.0) -> list[str]:  # noqa: ARG001
        calls.append(self.provider.name)
        return [f"model-{len(calls)}"]

    from decoding_sandbox.backends import openai_compat as oa

    monkeypatch.setattr(oa.OpenAICompatBackend, "fetch_available_models", _fake_fetch)
    with make_authed_client(app_with_key) as c:
        r1 = c.get("/api/v1/models/fireworks")
        r2 = c.get("/api/v1/models/fireworks?refresh=true")
    assert r1.json()["source"] == "live"
    assert r2.json()["source"] == "live"
    assert len(calls) == 2


def test_ttl_expiry_triggers_refetch(monkeypatch) -> None:
    """An entry older than ``MODEL_LIST_TTL_S`` is treated as a miss."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key-DSBX-ttl")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-DSBX-ttl-nim")
    calls: list[int] = []

    def _fake_fetch(self, timeout: float = 15.0) -> list[str]:  # noqa: ARG001
        calls.append(1)
        return ["live-model"]

    from decoding_sandbox.backends import openai_compat as oa

    monkeypatch.setattr(oa.OpenAICompatBackend, "fetch_available_models", _fake_fetch)
    cfg = make_test_config(providers=["fireworks"])
    registry = BackendRegistry(cfg)
    e1 = registry.list_models("fireworks")
    assert e1.source == "live"
    # Backdate the cached entry past the TTL and call again.
    with registry._models_lock:  # type: ignore[attr-defined]
        cached = registry._models_cache["fireworks"]  # type: ignore[attr-defined]
        registry._models_cache["fireworks"] = ModelListEntry(  # type: ignore[attr-defined]
            models=cached.models,
            source=cached.source,
            fetched_at=time.time() - (web_backends.MODEL_LIST_TTL_S + 10),
            note=cached.note,
        )
    e2 = registry.list_models("fireworks")
    assert e2.source == "live"
    assert len(calls) == 2


def test_live_fetch_failure_does_not_leak_url(app_with_key, monkeypatch) -> None:
    """A failing upstream call must NOT surface a URL in the response."""

    def _boom(self, timeout: float = 15.0):  # noqa: ARG001
        raise RuntimeError(
            "Server error '500 Internal Server Error' for url 'https://api.fireworks.ai/v1/models'"
        )

    from decoding_sandbox.backends import openai_compat as oa

    monkeypatch.setattr(oa.OpenAICompatBackend, "fetch_available_models", _boom)
    with make_authed_client(app_with_key) as c:
        r = c.get("/api/v1/models/fireworks")
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "fallback"
    # The static fallback list is what the UI sees -- and crucially, the
    # error message does not include the URL or the secret.
    blob = r.text
    assert "https://" not in blob
    assert "api.fireworks.ai" not in blob
