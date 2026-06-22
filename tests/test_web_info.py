"""``/api/v1/info`` listing + no-secrets-leak invariant.

The whole point of the middleware is that the browser never sees:

- ``base_url`` for any remote dsbx server (the dsbx-host LAN address).
- ``api_key_env`` or any actual API key for cloud providers.
- The ``secrets_env_file`` path.
- Any literal API key value present in the environment.

We assert all four by substring-scanning the serialized JSON. The test
deliberately uses values that would be obviously identifiable in a leak
(a non-RFC-1918-looking host string, a tagged secret value), so a future
regression would fail with a clear message.
"""

from __future__ import annotations

import json

import pytest

from decoding_sandbox.web.app import make_web_app
from tests.fakes import FakeBackend
from tests.web_helpers import (
    DEFAULT_TOKEN,
    build_test_app,
    make_authed_client,
    make_test_config,
)

SECRET_VALUE = "DSBX-TEST-SUPER-SECRET-VALUE-MUST-NOT-LEAK"


@pytest.fixture
def env_with_secret(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", SECRET_VALUE)
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET_VALUE + "-NIM")
    yield


@pytest.fixture
def app(env_with_secret):
    backend = FakeBackend(
        tokens={"ab": [97, 98]},
        pieces={97: "a", 98: "b"},
        distributions={},
        eos_token_ids=(99,),
    )
    cfg = make_test_config(
        secrets_env_file="/home/operator/.dsbx-secrets-test.env",
        remotes={"dsbx-host-py": "http://192.0.2.42:8000"},
        providers=["fireworks", "nim"],
    )
    return build_test_app({"dsbx-host-py": backend}, cfg=cfg)


def _scan(payload: object, needles: list[str]) -> list[str]:
    """Return the list of needles found anywhere in the serialized payload."""
    blob = json.dumps(payload)
    return [n for n in needles if n in blob]


def test_info_lists_remote_provider_local_backends(app) -> None:
    with make_authed_client(app) as c:
        r = c.get("/api/v1/info")
    assert r.status_code == 200
    data = r.json()
    assert data["default_backend"] == "dsbx-host-py"
    names = {b["name"] for b in data["backends"]}
    # Remote, cloud, and local entries should all be present.
    assert "dsbx-host-py" in names
    assert "fireworks" in names
    assert "nim" in names
    # The three built-in local engines are listed unconditionally.
    assert {"hf", "llamacpp", "llamacpp-py"} <= names


def test_info_reports_static_caps_for_unloaded_cloud_backends(app) -> None:
    """Cloud backends advertise their capability envelope BEFORE first use.

    The UI clamps inputs like ``alternatives (top-k)`` to
    ``capabilities.max_top_logprobs``. Cloud backends are lazy-loaded, so
    without synthesizing caps from the static :class:`ProviderConfig` the
    listing returns ``capabilities=null`` and the UI falls back to a
    generic max -- letting users set top_k=50 on Fireworks (capped to 5
    upstream). This test pins the static caps so a future refactor
    can't silently regress to "no caps until loaded".
    """
    with make_authed_client(app) as c:
        r = c.get("/api/v1/info")
    by_name = {b["name"]: b for b in r.json()["backends"]}

    fw = by_name["fireworks"]
    assert fw["capabilities"] is not None
    # ``make_test_config`` pins all provider caps to 5 to keep tests stable;
    # the production config differentiates (Fireworks=5, NIM/OpenRouter=20,
    # LM Studio=10) and the same plumbing surfaces those values too.
    assert fw["capabilities"]["max_top_logprobs"] == 5
    # Test fixture sets ``supports_prompt_logprobs=True`` only for fireworks.
    assert fw["capabilities"]["prompt_logprobs"] is True
    assert fw["capabilities"]["full_vocab"] is False

    nim = by_name["nim"]
    assert nim["capabilities"] is not None
    assert nim["capabilities"]["max_top_logprobs"] == 5
    assert nim["capabilities"]["prompt_logprobs"] is False


def test_info_marks_cloud_without_key_as_unavailable(env_with_secret, monkeypatch) -> None:
    # Build a fresh app where NIM has no key set.
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    backend = FakeBackend(tokens={}, pieces={}, distributions={}, eos_token_ids=(99,))
    cfg = make_test_config(providers=["fireworks", "nim"])
    app = build_test_app({"dsbx-host-py": backend}, cfg=cfg)
    with make_authed_client(app) as c:
        r = c.get("/api/v1/info")
    by_name = {b["name"]: b for b in r.json()["backends"]}
    assert by_name["nim"]["available"] is False
    assert "NVIDIA_API_KEY" in by_name["nim"]["note"]
    # Fireworks has a key from the fixture, so it stays available.
    assert by_name["fireworks"]["available"] is True


def test_info_does_not_leak_base_urls_or_keys(app) -> None:
    with make_authed_client(app) as c:
        r = c.get("/api/v1/info")
    payload = r.json()
    leaked = _scan(
        payload,
        [
            # Remote base_url + bare host of dsbx-host.
            "http://192.0.2.42:8000",
            "192.0.2.42",
            # Provider base URLs (set by make_test_config).
            "https://api.example/fireworks",
            "https://api.example/nim",
            # The secret value bound to FIREWORKS_API_KEY.
            SECRET_VALUE,
            # The configured secrets file path.
            "/home/operator/.dsbx-secrets-test.env",
            # The api_key_env *name* must also be absent so the UI can't
            # be tempted to read it from window state.
            "FIREWORKS_API_KEY",
            "NVIDIA_API_KEY",
        ],
    )
    assert leaked == [], f"info leaked: {leaked}"


def test_info_default_backend_is_a_logical_name_not_a_url(app) -> None:
    with make_authed_client(app) as c:
        r = c.get("/api/v1/info")
    default = r.json()["default_backend"]
    assert not default.startswith("http")
    assert "://" not in default


def test_health_payload_contains_no_secrets(app) -> None:
    """Health is unauthenticated -- it had better not leak anything either."""
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        r = c.get("/api/v1/health")
    assert r.status_code == 200
    body = r.text
    assert "192.0.2.42" not in body
    assert SECRET_VALUE not in body
    assert DEFAULT_TOKEN not in body


def test_construction_rejects_empty_token(env_with_secret) -> None:
    cfg = make_test_config()
    with pytest.raises(ValueError, match="non-empty"):
        make_web_app(cfg, token="")


def test_info_exposes_loaded_model_and_suggestions(app) -> None:
    """The browser needs ``loaded_model`` + ``suggested_models`` to render a
    proper model picker; cloud providers are the only ``model_editable``
    family today."""
    with make_authed_client(app) as c:
        r = c.get("/api/v1/info")
    rows = {b["name"]: b for b in r.json()["backends"]}
    # Cloud providers: editable, advertise their default and any extras.
    fw = rows["fireworks"]
    assert fw["model_editable"] is True
    assert fw["loaded_model"] == "fireworks/default-model"
    assert fw["loaded_model"] in fw["suggested_models"]
    nim = rows["nim"]
    assert nim["model_editable"] is True
    assert nim["loaded_model"] == "nim/default-model"
    # Remote: not editable, listing carries no model until the upstream is
    # actually contacted (the test app never hits the wire so loaded_model
    # stays null here; the UI shows "unknown until loaded").
    wp = rows["dsbx-host-py"]
    assert wp["model_editable"] is False
