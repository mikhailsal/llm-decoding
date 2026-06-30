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

from dsbx.web.app import make_web_app
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


def test_info_static_caps_mirror_all_provider_supports_flags(env_with_secret) -> None:
    """Static-caps stub (used before the backend is loaded) must expose
    EVERY ``supports_*`` flag declared on :class:`ProviderConfig`.

    The Chrome MCP manual check caught a regression where Fireworks'
    ``respect EOS`` checkbox stayed locked, ``service tier`` dropdown
    stayed hidden, and ``logit_bias`` editor refused input until the
    first generate call lazily-loaded the real backend. Cause: the
    stub in ``web/backends.py`` only forwarded ``prompt_logprobs`` and
    ``max_top_logprobs`` -- the seven new flags from Phases 1-5 fell
    back to ``False``. This pin makes sure the stub stays in sync with
    any future addition to ``ProviderConfig``.
    """
    from dsbx.core.config import (
        Config,
        ProviderConfig,
        RemoteConfig,
        StorageConfig,
    )

    cfg = Config(
        raw={
            "secrets_env_file": "/tmp/dsbx-test-secrets-DO-NOT-USE.env",
            "run": {"backend": "dsbx-host-py"},
            "storage": {
                "hf_home": "/tmp/hf",
                "pip_cache": "/tmp/pip",
                "min_free_gb": 1.0,
                "check_paths": ["/tmp"],
            },
            "local": {},
            "remote": {"dsbx-host-py": {"base_url": "http://192.0.2.42:8000"}},
            "providers": {},
            "web": {"api_token": DEFAULT_TOKEN, "logging": {"enabled": False}},
        },
        config_path=None,
        secrets_env_file="/tmp/dsbx-test-secrets-DO-NOT-USE.env",
        default_backend="dsbx-host-py",
        storage=StorageConfig(
            hf_home="/tmp/hf",
            pip_cache="/tmp/pip",
            min_free_gb=1.0,
            check_paths=["/tmp"],
        ),
        providers={
            "fireworks": ProviderConfig(
                name="fireworks",
                base_url="https://api.example/fireworks",
                api_key_env="FIREWORKS_API_KEY",
                default_model="fireworks/x",
                max_top_logprobs=5,
                supports_prompt_logprobs=True,
                supports_new_logprobs=True,
                supports_sampling_mask=True,
                supports_ignore_eos=True,
                supports_perf_metrics=True,
                supports_service_tier=True,
                supports_raw_output=True,
                supports_logit_bias=True,
                supports_combined_echo_stream=True,
                has_completions=True,
            )
        },
        remotes={
            "dsbx-host-py": RemoteConfig(
                name="dsbx-host-py", base_url="http://192.0.2.42:8000", timeout=10.0
            )
        },
    )
    backend = FakeBackend(
        tokens={"x": [120]}, pieces={120: "x"}, distributions={}, eos_token_ids=(99,)
    )
    app = build_test_app({"dsbx-host-py": backend}, cfg=cfg)
    with make_authed_client(app) as c:
        r = c.get("/api/v1/info")
    by_name = {b["name"]: b for b in r.json()["backends"]}
    caps = by_name["fireworks"]["capabilities"]
    assert caps is not None
    # Every flag below must be ``True`` because the stub MUST mirror what
    # the loaded OpenAICompatBackend would advertise. ``can_force_token``
    # comes from ``has_completions`` (Fireworks /v1/completions allows
    # forcing).
    assert caps["supports_ignore_eos"] is True
    assert caps["supports_perf_metrics"] is True
    assert caps["supports_service_tier"] is True
    assert caps["supports_sampling_mask"] is True
    assert caps["supports_raw_output"] is True
    assert caps["supports_logit_bias"] is True
    assert caps["supports_combined_echo_stream"] is True
    assert caps["can_force_token"] is True
    assert caps["notes"] == "static caps from provider config (backend not yet loaded)"


def test_info_models_caps_carry_per_model_overrides_before_load(
    env_with_secret,
) -> None:
    """Synthetic per-model envelopes honour ``ProviderConfig.model_overrides``.

    Before ANY backend instance is loaded (so ``cloud_variants`` is
    empty) the listing must already publish a ``models_caps`` map
    with one entry per curated model, and each entry's ``supports_*``
    flags must reflect ``model_overrides``. Without this the UI would
    only learn that ``gpt-oss-20b`` rejects ``sampling_mask`` AFTER
    the first generate call lands with a confusing 400 ("Extra inputs
    are not permitted"). With it, the eligible-after-filters column
    and the ``sampling_mask`` knob are gated correctly from page load.
    """
    from dsbx.core.config import (
        Config,
        ProviderConfig,
        RemoteConfig,
        StorageConfig,
    )

    cfg = Config(
        raw={
            "secrets_env_file": "/tmp/dsbx-test-secrets-DO-NOT-USE.env",
            "run": {"backend": "dsbx-host-py"},
            "storage": {
                "hf_home": "/tmp/hf",
                "pip_cache": "/tmp/pip",
                "min_free_gb": 1.0,
                "check_paths": ["/tmp"],
            },
            "local": {},
            "remote": {"dsbx-host-py": {"base_url": "http://192.0.2.42:8000"}},
            "providers": {},
            "web": {"api_token": DEFAULT_TOKEN, "logging": {"enabled": False}},
        },
        config_path=None,
        secrets_env_file="/tmp/dsbx-test-secrets-DO-NOT-USE.env",
        default_backend="dsbx-host-py",
        storage=StorageConfig(
            hf_home="/tmp/hf",
            pip_cache="/tmp/pip",
            min_free_gb=1.0,
            check_paths=["/tmp"],
        ),
        providers={
            "fireworks": ProviderConfig(
                name="fireworks",
                base_url="https://api.example/fireworks",
                api_key_env="FIREWORKS_API_KEY",
                default_model="acct/models/gpt-oss-120b",
                max_top_logprobs=5,
                has_completions=True,
                supports_prompt_logprobs=True,
                supports_new_logprobs=True,
                supports_sampling_mask=True,
                models=[
                    "acct/models/gpt-oss-120b",
                    "acct/models/gpt-oss-20b",
                    "acct/models/glm-5p1",
                ],
                tokenizers={
                    "acct/models/gpt-oss-120b": "openai/gpt-oss-120b",
                    "acct/models/gpt-oss-20b": "openai/gpt-oss-20b",
                    # No mapping for glm-5p1 -> supports_local_tokenize=False
                },
                model_overrides={
                    "acct/models/gpt-oss-20b": {"supports_sampling_mask": False},
                },
            )
        },
        remotes={
            "dsbx-host-py": RemoteConfig(
                name="dsbx-host-py", base_url="http://192.0.2.42:8000", timeout=10.0
            )
        },
    )
    backend = FakeBackend(
        tokens={"x": [120]},
        pieces={120: "x"},
        distributions={},
        eos_token_ids=(99,),
    )
    app = build_test_app({"dsbx-host-py": backend}, cfg=cfg)
    with make_authed_client(app) as c:
        r = c.get("/api/v1/info")
    by_name = {b["name"]: b for b in r.json()["backends"]}
    fw = by_name["fireworks"]
    assert fw["models_caps"], (
        "expected models_caps to be pre-populated for curated cloud models, got empty map"
    )
    caps_120 = fw["models_caps"]["acct/models/gpt-oss-120b"]
    caps_20 = fw["models_caps"]["acct/models/gpt-oss-20b"]
    caps_glm = fw["models_caps"]["acct/models/glm-5p1"]
    # Override fires only for gpt-oss-20b.
    assert caps_120["supports_sampling_mask"] is True
    assert caps_20["supports_sampling_mask"] is False
    # glm-5p1 has no tokenizer mapping -> local tokenize off; the
    # other two have mappings -> local tokenize on (so the live
    # preview works).
    assert caps_120["supports_local_tokenize"] is True
    assert caps_20["supports_local_tokenize"] is True
    assert caps_glm["supports_local_tokenize"] is False
    # Top-level capabilities still mirrors the default model (back-compat).
    assert fw["capabilities"]["supports_sampling_mask"] is True


def test_info_marks_chat_only_providers_generation_disabled(env_with_secret) -> None:
    """NIM (chat-only -- ``has_completions=false``) advertises
    ``generation_disabled=true`` in the static caps stub returned by
    ``/api/v1/info``, plus a tooltip-ready note in ``capabilities.notes``.

    The frontend backend picker uses these to render the option as
    ``<option disabled title="chat-only...">``. The route guard in
    ``/api/v1/generate/stream`` is the authoritative gate; this flag
    is the pre-flight UX so the user sees the decision before clicking.
    Fireworks (``has_completions=true``) must stay enabled.
    """
    backend = FakeBackend(tokens={}, pieces={}, distributions={}, eos_token_ids=(99,))
    cfg = make_test_config(providers=["fireworks", "nim"])
    app = build_test_app({"dsbx-host-py": backend}, cfg=cfg)
    with make_authed_client(app) as c:
        r = c.get("/api/v1/info")
    by_name = {b["name"]: b for b in r.json()["backends"]}

    nim_caps = by_name["nim"]["capabilities"]
    assert nim_caps is not None
    assert nim_caps["generation_disabled"] is True
    assert "chat-only" in nim_caps["notes"]

    fw_caps = by_name["fireworks"]["capabilities"]
    assert fw_caps is not None
    assert fw_caps["generation_disabled"] is False


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
