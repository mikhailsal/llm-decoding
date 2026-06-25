"""``/api/v1/backends/{name}/status`` + ``/reload``: remote model control.

These two endpoints let the browser drive the swappable model slot on a
remote ``dsbx serve`` host. They must:

- proxy the upstream status/reload faithfully (state, loaded_model, error);
- forward the requested model id to the host's ``reload_model``;
- guard against non-remote families (cloud/local) with a clean 400;
- 404 an unknown backend;
- never leak the dsbx-host LAN address / base_url in any error.

We stand in a ``FakeRemote`` for the real ``RemoteBackend`` so the test
stays on the wire-free path (no httpx, no dsbx-host), while still exercising
the registry plumbing, the family guard, and the auth dependency.
"""

from __future__ import annotations

from tests.fakes import FakeBackend
from decoding_sandbox.web.backends import BackendRegistry

from tests.web_helpers import build_test_app, make_authed_client, make_test_config


class FakeRemote(FakeBackend):
    """A FakeBackend that also speaks the remote model-slot control API."""

    def __init__(self) -> None:
        super().__init__(tokens={}, pieces={}, distributions={}, eos_token_ids=(99,))
        self._status = {"state": "ready", "loaded_model": "m.gguf", "error": None}
        self.reload_calls: list[str | None] = []
        self.unload_calls = 0
        self.refreshed = 0

    def server_status(self) -> dict:
        return dict(self._status)

    def list_server_models(self) -> list[dict]:
        return [
            {"id": "/m/a.gguf", "label": "a"},
            {"id": "/m/b.gguf", "label": "b"},
        ]

    def reload_model(self, model: str | None) -> dict:
        self.reload_calls.append(model)
        self._status = {"state": "loading", "loaded_model": None, "error": None}
        return dict(self._status)

    def unload_model(self) -> dict:
        self.unload_calls += 1
        self._status = {"state": "empty", "loaded_model": None, "error": None}
        return dict(self._status)

    def refresh_info(self) -> None:
        self.refreshed += 1


def _app_with_remote() -> tuple:
    fr = FakeRemote()
    cfg = make_test_config(
        remotes={"dsbx-host-py": "http://192.0.2.42:8000"},
        providers=["fireworks", "nim"],
    )
    return build_test_app({"dsbx-host-py": fr}, cfg=cfg), fr


def test_remote_status_proxies_upstream() -> None:
    app, fr = _app_with_remote()
    with make_authed_client(app) as c:
        r = c.get("/api/v1/backends/dsbx-host-py/status")
    assert r.status_code == 200
    data = r.json()
    assert data["backend"] == "dsbx-host-py"
    assert data["state"] == "ready"
    assert data["loaded_model"] == "m.gguf"
    # ``ready`` triggers a capability refresh so /info reflects the model.
    assert fr.refreshed == 1


def test_remote_reload_forwards_model_and_returns_loading() -> None:
    app, fr = _app_with_remote()
    with make_authed_client(app) as c:
        r = c.post("/api/v1/backends/dsbx-host-py/reload", json={"model": "/m/b.gguf"})
    assert r.status_code == 200
    assert fr.reload_calls == ["/m/b.gguf"]
    assert r.json()["state"] == "loading"


def test_registry_unload_forwards_and_returns_empty() -> None:
    fr = FakeRemote()
    registry = BackendRegistry(
        make_test_config(remotes={"dsbx-host-py": "http://192.0.2.42:8000"})
    )
    registry.get("dsbx-host-py").instance = fr
    registry._models_cache["dsbx-host-py"] = object()  # type: ignore[assignment]

    data = registry.unload_remote("dsbx-host-py")

    assert fr.unload_calls == 1
    assert fr.refreshed == 1
    assert data["state"] == "empty"
    assert data["loaded_model"] is None
    assert "dsbx-host-py" not in registry._models_cache  # type: ignore[attr-defined]


def test_remote_models_lists_live_catalogue() -> None:
    app, _fr = _app_with_remote()
    with make_authed_client(app) as c:
        r = c.get("/api/v1/models/dsbx-host-py")
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "live"
    assert data["models"] == ["/m/a.gguf", "/m/b.gguf"]


def test_status_400_for_non_remote_backend() -> None:
    app, _fr = _app_with_remote()
    with make_authed_client(app) as c:
        r = c.get("/api/v1/backends/fireworks/status")
    assert r.status_code == 400
    # The guard message must not leak the provider's base_url.
    assert "https://" not in r.json()["detail"]


def test_reload_404_for_unknown_backend() -> None:
    app, _fr = _app_with_remote()
    with make_authed_client(app) as c:
        r = c.post("/api/v1/backends/nope/reload", json={"model": None})
    assert r.status_code == 404


def test_registry_unload_unknown_backend_raises_lookup_error() -> None:
    registry = BackendRegistry(
        make_test_config(remotes={"dsbx-host-py": "http://192.0.2.42:8000"})
    )
    try:
        registry.unload_remote("nope")
    except LookupError:
        pass
    else:  # pragma: no cover - defensive assertion style for old pytest.
        raise AssertionError("expected LookupError")


def test_status_requires_auth() -> None:
    app, _fr = _app_with_remote()
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        r = c.get("/api/v1/backends/dsbx-host-py/status")
    assert r.status_code == 401
