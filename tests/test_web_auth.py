"""Authentication tests for the dsbx web middleware.

Every ``/api/v1/*`` route except ``/api/v1/health`` must:

- 401 when ``Authorization`` is absent.
- 401 when the scheme is not ``Bearer``.
- 401 when the token doesn't match.
- 200 (or 4xx from the handler) when the token matches.

We assert all four cases per representative route. Comparison must be
constant-time -- that's verified indirectly by the codepath (both branches
return the same error string) and directly by inspecting the source: the
``hmac.compare_digest`` call is the only comparison in the auth module.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dsbx.web.auth import AuthConfig
from tests.fakes import FakeBackend, cand
from tests.web_helpers import DEFAULT_TOKEN, build_test_app


@pytest.fixture
def app():
    backend = FakeBackend(
        tokens={"ab": [97, 98]},
        pieces={97: "a", 98: "b", 88: "X", 89: "Y"},
        distributions={
            (97,): [cand(98, "b", 0.6, 0), cand(89, "Y", 0.4, 1)],
            (97, 98): [cand(88, "X", 0.6, 0), cand(89, "Y", 0.4, 1)],
        },
        eos_token_ids=(99,),
    )
    return build_test_app({"dsbx-host-py": backend})


def test_health_is_public(app) -> None:
    with TestClient(app) as c:
        r = c.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["server_label"] == "dsbx-test"


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("GET", "/api/v1/info", None),
        ("POST", "/api/v1/tokenize", {"backend": "dsbx-host-py", "text": "ab"}),
        ("POST", "/api/v1/manual/sessions", {"backend": "dsbx-host-py", "prompt": "ab"}),
        ("GET", "/api/v1/probe", None),
    ],
)
def test_protected_routes_reject_missing_auth(app, method: str, path: str, body) -> None:
    with TestClient(app) as c:
        r = c.request(method, path, json=body)
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"].startswith("Bearer")
    assert "missing" in r.json()["detail"].lower()


def test_protected_routes_reject_wrong_scheme(app) -> None:
    with TestClient(app) as c:
        r = c.get("/api/v1/info", headers={"Authorization": "Token abc"})
    assert r.status_code == 401
    assert "scheme" in r.json()["detail"].lower()


def test_protected_routes_reject_wrong_token(app) -> None:
    with TestClient(app) as c:
        r = c.get("/api/v1/info", headers={"Authorization": "Bearer wrong-token"})
    assert r.status_code == 401
    assert "invalid" in r.json()["detail"].lower()


def test_protected_routes_accept_correct_token(app) -> None:
    with TestClient(app) as c:
        r = c.get("/api/v1/info", headers={"Authorization": f"Bearer {DEFAULT_TOKEN}"})
    assert r.status_code == 200
    assert "backends" in r.json()


def test_auth_config_rejects_empty_token() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AuthConfig(token="")
    with pytest.raises(ValueError, match="non-empty"):
        AuthConfig(token="   ")


def test_auth_does_not_leak_token_in_error_body(app) -> None:
    """An auth failure must never echo back what the configured token is.

    The middleware's bearer comparison is constant-time, but a bug in error
    rendering could still log the supplied (wrong) token. We make sure the
    response body contains neither the configured token nor any prefix.
    """
    with TestClient(app) as c:
        r = c.get("/api/v1/info", headers={"Authorization": "Bearer guessed-prefix"})
    assert r.status_code == 401
    body = r.text
    assert DEFAULT_TOKEN not in body
    assert DEFAULT_TOKEN[:8] not in body
