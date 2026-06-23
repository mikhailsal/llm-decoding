"""``/api/v1/special_tokens`` -- the Decode composer's palette source.

The endpoint exposes a backend's special / added tokens so the browser can
render a per-model "insert special token" palette. We assert the happy path
(tokens flow through, shape is ``{id, text}``) and the empty path (a backend
with no introspectable tokenizer returns an empty list rather than erroring).
"""

from __future__ import annotations

from tests.fakes import FakeBackend
from tests.web_helpers import build_test_app, make_authed_client


def test_special_tokens_endpoint_returns_palette() -> None:
    backend = FakeBackend(
        special_tokens=[(2, "<|endoftext|>"), (1, "<|startoftext|>")],
    )
    app = build_test_app({"dsbx-host-py": backend})
    client = make_authed_client(app)

    r = client.post("/api/v1/special_tokens", json={"backend": "dsbx-host-py"})
    assert r.status_code == 200
    tokens = r.json()["tokens"]
    assert {"id": 2, "text": "<|endoftext|>"} in tokens
    assert {"id": 1, "text": "<|startoftext|>"} in tokens


def test_special_tokens_endpoint_empty_for_backend_without_palette() -> None:
    backend = FakeBackend()  # special_tokens defaults to []
    app = build_test_app({"dsbx-host-py": backend})
    client = make_authed_client(app)

    r = client.post("/api/v1/special_tokens", json={"backend": "dsbx-host-py"})
    assert r.status_code == 200
    assert r.json() == {"tokens": []}
