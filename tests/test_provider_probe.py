"""Tests for the live provider-probe with HTTP mocked out."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from decoding_sandbox.core import provider_probe
from decoding_sandbox.core.config import ProviderConfig


def _prov(
    name: str = "fireworks",
    *,
    supports_prompt_logprobs: bool = True,
    has_completions: bool = True,
    require_parameters: bool = False,
    max_top: int = 5,
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        base_url=f"https://{name}.example/v1",
        api_key_env=f"{name.upper()}_API_KEY",
        default_model=f"{name}/model",
        max_top_logprobs=max_top,
        supports_prompt_logprobs=supports_prompt_logprobs,
        require_parameters=require_parameters,
        has_completions=has_completions,
    )


class _PostRecorder:
    """Stand-in for ``httpx.post`` that returns canned responses."""

    def __init__(self, routes: dict[str, dict[str, Any]], status_overrides: dict[str, int] | None = None) -> None:
        self.routes = routes
        self.status_overrides = status_overrides or {}
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> Any:
        self.calls.append({"url": url, "headers": headers, "json": json})
        for suffix, body in self.routes.items():
            if url.endswith(suffix):
                status = self.status_overrides.get(suffix, 200)
                return _FakeHTTPResponse(status, body)
        raise AssertionError(f"unregistered URL {url}")


class _FakeHTTPResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_probe_chat_returns_alt_count_when_logprobs_present(monkeypatch) -> None:
    prov = _prov()
    recorder = _PostRecorder({
        "/chat/completions": {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "Paris", "logprob": -0.5, "top_logprobs": [{"token": "Paris", "logprob": -0.5}, {"token": "London", "logprob": -3.0}]}
                        ]
                    }
                }
            ]
        }
    })
    monkeypatch.setattr(provider_probe.httpx, "post", recorder)

    out = provider_probe._probe_chat(prov, "test-key", "fireworks/model")

    assert out == "ok (2 alts)"
    assert recorder.calls[0]["json"]["top_logprobs"] == 5
    assert recorder.calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_probe_chat_reports_no_logprobs_when_content_empty(monkeypatch) -> None:
    prov = _prov()
    monkeypatch.setattr(
        provider_probe.httpx, "post",
        _PostRecorder({"/chat/completions": {"choices": [{"logprobs": None}]}}),
    )

    assert provider_probe._probe_chat(prov, "k", "m") == "no logprobs field"


def test_probe_chat_passes_provider_require_parameters(monkeypatch) -> None:
    prov = _prov("openrouter", require_parameters=True, supports_prompt_logprobs=False, has_completions=False, max_top=20)
    recorder = _PostRecorder({
        "/chat/completions": {"choices": [{"logprobs": {"content": [{"top_logprobs": []}]}}]}
    })
    monkeypatch.setattr(provider_probe.httpx, "post", recorder)

    provider_probe._probe_chat(prov, "k", "m")

    body = recorder.calls[0]["json"]
    assert body["provider"] == {"require_parameters": True}


def test_probe_chat_reports_http_status_error(monkeypatch) -> None:
    prov = _prov()
    monkeypatch.setattr(
        provider_probe.httpx, "post",
        _PostRecorder({"/chat/completions": {}}, status_overrides={"/chat/completions": 401}),
    )

    assert provider_probe._probe_chat(prov, "k", "m") == "err: HTTP 401"


def test_probe_chat_reports_connection_error(monkeypatch) -> None:
    prov = _prov()

    def boom(*a, **kw):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(provider_probe.httpx, "post", boom)
    assert provider_probe._probe_chat(prov, "k", "m") == "err: ConnectError"


def test_probe_chat_reports_invalid_json(monkeypatch) -> None:
    prov = _prov()
    monkeypatch.setattr(
        provider_probe.httpx, "post",
        _PostRecorder({"/chat/completions": ValueError("not json")}),
    )
    assert provider_probe._probe_chat(prov, "k", "m") == "err: non-JSON response"


def test_probe_prompt_skipped_when_provider_unsupported() -> None:
    prov = _prov(supports_prompt_logprobs=False)
    assert provider_probe._probe_prompt(prov, "k", "m") == "n/a"


def test_probe_prompt_ok_when_tokens_returned(monkeypatch) -> None:
    prov = _prov()
    monkeypatch.setattr(
        provider_probe.httpx, "post",
        _PostRecorder({
            "/completions": {
                "choices": [
                    {"logprobs": {"tokens": ["The", " cap", "ital"], "token_logprobs": [None, -1.0, -2.0]}}
                ]
            }
        }),
    )
    assert provider_probe._probe_prompt(prov, "k", "m") == "ok (3 tokens)"


def test_probe_prompt_reports_404(monkeypatch) -> None:
    prov = _prov()
    monkeypatch.setattr(
        provider_probe.httpx, "post",
        _PostRecorder({"/completions": {}}, status_overrides={"/completions": 404}),
    )
    assert provider_probe._probe_prompt(prov, "k", "m") == "no (/completions 404)"


def test_probe_prompt_reports_missing_tokens(monkeypatch) -> None:
    prov = _prov()
    monkeypatch.setattr(
        provider_probe.httpx, "post",
        _PostRecorder({"/completions": {"choices": [{"logprobs": {}}]}}),
    )
    assert provider_probe._probe_prompt(prov, "k", "m") == "no prompt logprobs"


def test_probe_provider_short_circuits_without_api_key(monkeypatch) -> None:
    prov = _prov()
    monkeypatch.delenv(prov.api_key_env, raising=False)

    result = provider_probe.probe_provider(prov, model="m")

    assert "no API key" in result.chat_logprobs
    assert "no API key" in result.prompt_logprobs


def test_run_probe_renders_table_and_returns_zero_on_success(monkeypatch) -> None:
    from decoding_sandbox.core.config import Config, StorageConfig

    prov = _prov()
    cfg = Config(
        raw={},
        config_path=None,
        secrets_env_file="",
        default_backend="llamacpp",
        storage=StorageConfig(hf_home="", pip_cache="", min_free_gb=5.0, check_paths=[]),
        providers={"fireworks": prov},
    )

    def fake_probe(p, model):
        return provider_probe.ProbeResult(
            provider=p.name, model=model or p.default_model,
            chat_logprobs="ok (2 alts)", prompt_logprobs="ok (3 tokens)",
        )

    monkeypatch.setattr(provider_probe, "probe_provider", fake_probe)

    from rich.console import Console
    import io

    output = io.StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)

    rc = provider_probe.run_probe(cfg, providers=["fireworks"], model=None, console=console)

    assert rc == 0
    rendered = output.getvalue()
    assert "fireworks" in rendered
    assert "ok (2 alts)" in rendered


def test_run_probe_returns_one_when_any_chat_errors(monkeypatch) -> None:
    from decoding_sandbox.core.config import Config, StorageConfig

    prov = _prov()
    cfg = Config(
        raw={}, config_path=None, secrets_env_file="",
        default_backend="llamacpp",
        storage=StorageConfig(hf_home="", pip_cache="", min_free_gb=5.0, check_paths=[]),
        providers={"fireworks": prov},
    )

    monkeypatch.setattr(
        provider_probe, "probe_provider",
        lambda p, model: provider_probe.ProbeResult(p.name, "m", "err: HTTP 500", "n/a"),
    )

    rc = provider_probe.run_probe(cfg, providers=["fireworks"], model=None, console=None)
    assert rc == 1


def test_run_probe_returns_two_when_provider_unknown(monkeypatch) -> None:
    from decoding_sandbox.core.config import Config, StorageConfig

    cfg = Config(
        raw={}, config_path=None, secrets_env_file="", default_backend="llamacpp",
        storage=StorageConfig(hf_home="", pip_cache="", min_free_gb=5.0, check_paths=[]),
        providers={},
    )
    import io
    from rich.console import Console
    out = io.StringIO()
    rc = provider_probe.run_probe(
        cfg, providers=["nope"], model=None,
        console=Console(file=out, force_terminal=False, color_system=None, width=80),
    )
    assert rc == 2


@pytest.mark.parametrize("require_parameters", [True, False])
def test_probe_chat_sets_or_skips_provider_field(monkeypatch, require_parameters: bool) -> None:
    prov = _prov(require_parameters=require_parameters)
    recorder = _PostRecorder({
        "/chat/completions": {"choices": [{"logprobs": {"content": [{"top_logprobs": []}]}}]}
    })
    monkeypatch.setattr(provider_probe.httpx, "post", recorder)

    provider_probe._probe_chat(prov, "k", "m")

    body = recorder.calls[0]["json"]
    if require_parameters:
        assert body.get("provider") == {"require_parameters": True}
    else:
        assert "provider" not in body
