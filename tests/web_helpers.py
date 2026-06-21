"""Shared test helpers for the dsbx web middleware.

The web middleware reaches into :mod:`decoding_sandbox.core.factory` to build
backends, but a test wants to short-circuit that and inject a pre-canned
:class:`FakeBackend`. We do it via :func:`build_test_app`: it constructs a
real ``FastAPI`` app from :func:`make_web_app` and then monkeypatches the
registry so a fixed-name backend yields a fixed instance.

That keeps the test path *just* indirect enough that we exercise the auth
dependency, the registry locking, and the full request/response cycle, but
without ever instantiating heavy engines or talking to dsbx-host.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from fastapi.testclient import TestClient

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import (
    Config,
    ProviderConfig,
    RemoteConfig,
    StorageConfig,
)
from decoding_sandbox.web.app import make_web_app
from decoding_sandbox.web.backends import BackendRegistry, _BackendEntry

DEFAULT_TOKEN = "test-token-please-rotate-1234567890"

# Matches the real-world env-var names used by config.example.toml so tests
# referencing FIREWORKS_API_KEY / NVIDIA_API_KEY / etc. exercise the same
# path production does. Anything not listed here falls back to <NAME>_API_KEY.
_PROVIDER_ENV = {
    "fireworks": "FIREWORKS_API_KEY",
    "nim": "NVIDIA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "lmstudio": "LMSTUDIO_API_KEY",
}


def make_test_config(
    *,
    secrets_env_file: str = "/tmp/dsbx-test-secrets-DO-NOT-USE.env",
    remotes: dict[str, str] | None = None,
    providers: list[str] | None = None,
    web_token: str = DEFAULT_TOKEN,
) -> Config:
    """Build a minimal Config that looks plausible to the registry.

    ``remotes`` maps logical names to fake base URLs; ``providers`` lists
    names that get a stub ProviderConfig (with the env-key bound but
    unset, so the registry marks them unavailable -- exactly the path
    we want to assert on for cloud).
    """
    remotes = remotes or {"dsbx-host-py": "http://192.0.2.42:8000"}
    providers = providers or ["fireworks", "nim"]
    return Config(
        raw={
            "secrets_env_file": secrets_env_file,
            "run": {"backend": "dsbx-host-py"},
            "storage": {
                "hf_home": "/tmp/hf",
                "pip_cache": "/tmp/pip",
                "min_free_gb": 1.0,
                "check_paths": ["/tmp"],
            },
            "local": {},
            "remote": {n: {"base_url": u} for n, u in remotes.items()},
            "providers": {},
            "web": {"api_token": web_token},
        },
        config_path=None,
        secrets_env_file=secrets_env_file,
        default_backend="dsbx-host-py",
        storage=StorageConfig(
            hf_home="/tmp/hf",
            pip_cache="/tmp/pip",
            min_free_gb=1.0,
            check_paths=["/tmp"],
        ),
        providers={
            name: ProviderConfig(
                name=name,
                base_url=f"https://api.example/{name}",
                api_key_env=_PROVIDER_ENV.get(name, f"{name.upper()}_API_KEY"),
                default_model=f"{name}/default-model",
                max_top_logprobs=5,
                supports_prompt_logprobs=(name == "fireworks"),
            )
            for name in providers
        },
        remotes={
            name: RemoteConfig(name=name, base_url=url, timeout=10.0)
            for name, url in remotes.items()
        },
    )


@contextmanager
def patched_registry(app, backends: dict[str, Backend]) -> Iterator[BackendRegistry]:
    """Override the app's registry to return the given fakes.

    Iterates the existing entries -- so the public ``/api/v1/info`` listing
    keeps the same shape -- but replaces ``ensure_loaded`` so any of
    ``backends``' keys yields a fake instance. Anything not in ``backends``
    raises ``KeyError`` from the route's perspective (HTTP 404).
    """
    registry: BackendRegistry = app.state.registry

    # Inject the fakes as pre-loaded entries.
    for name, backend in backends.items():
        if name not in registry._entries:  # type: ignore[attr-defined]
            registry._entries[name] = _BackendEntry(  # type: ignore[attr-defined]
                name=name, family="remote"
            )
        registry._entries[name].instance = backend  # type: ignore[attr-defined]
        registry._entries[name].lock = threading.Lock()  # type: ignore[attr-defined]

    yield registry


def build_test_app(
    backends: dict[str, Backend],
    *,
    token: str = DEFAULT_TOKEN,
    cfg: Config | None = None,
    cors_origins: list[str] | None = None,
):
    """Build the FastAPI app with ``backends`` pre-installed.

    The returned object has both ``app`` (for direct inspection) and a
    convenience ``client`` (``TestClient``); enter ``client`` as a
    context manager to fire the FastAPI startup/shutdown events.
    """
    cfg = cfg or make_test_config(web_token=token)
    app = make_web_app(cfg, token=token, server_label="dsbx-test", cors_origins=cors_origins)
    # Replace registry entries with our fakes BEFORE any client request.
    registry: BackendRegistry = app.state.registry
    for name, backend in backends.items():
        if name not in registry._entries:  # type: ignore[attr-defined]
            registry._entries[name] = _BackendEntry(name=name, family="remote")  # type: ignore[attr-defined]
        registry._entries[name].instance = backend  # type: ignore[attr-defined]
    return app


def make_authed_client(app, token: str = DEFAULT_TOKEN) -> TestClient:
    """TestClient that injects the Bearer token on every request."""
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {token}"
    return client


__all__ = [
    "DEFAULT_TOKEN",
    "make_test_config",
    "build_test_app",
    "make_authed_client",
    "patched_registry",
]
