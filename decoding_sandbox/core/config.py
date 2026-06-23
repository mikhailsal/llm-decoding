"""Configuration loading for the decoding sandbox.

Reads ``config.toml`` (gitignored) if present, otherwise ``config.example.toml``,
and merges it over built-in defaults. Also handles loading a dotenv-style secrets
file so provider API keys (referenced by name via ``api_key_env``) are available
in ``os.environ`` without ever being stored in the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib  # type: ignore

# Repo root = two levels up from this file (decoding_sandbox/core/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Defaults (kept in sync with config.example.toml; the file overrides these).
# --------------------------------------------------------------------------- #
_DEFAULTS: dict[str, Any] = {
    "secrets_env_file": "~/.config/dsbx/secrets.env",
    "run": {"backend": "llamacpp"},
    "storage": {
        "hf_home": "~/.cache/dsbx/huggingface",
        "pip_cache": "~/.cache/dsbx/pip",
        "min_free_gb": 5.0,
        "check_paths": ["/", "~/.cache/dsbx", "~"],
    },
    "local": {
        "llamacpp": {
            "base_url": "http://127.0.0.1:8080",
            "model": "Qwen3.5-9B-Base-Q4_K_M",
        },
        "llamacpp_py": {
            # In-process llama.cpp via llama-cpp-python. Same GGUF as the HTTP
            # backend, but with `logits_all=True` so we expose the FULL [seq,
            # vocab] tensor (true white-box on Qwen3.5-9B, which HF can't load
            # on the 6 GB Pascal). model_path is auto-discovered from the HF
            # cache if left null.
            "model_path": None,
            "model_glob": "**/Qwen3.5-9B-Base-Q4_K_M.gguf",
            "model_search_dirs": [
                "~/.cache/dsbx/huggingface",
                "~/.cache/huggingface",
            ],
            "n_gpu_layers": 20,
            "n_ctx": 4096,
            "logits_all": True,
            "verbose": False,
        },
        "hf": {
            # 9B base doesn't load in 4-bit on the 6 GB Pascal (bnb/accelerate
            # meta-tensor bug on the hybrid arch); white-box uses a dense base,
            # 9B base is served by llama.cpp. See config.example.toml.
            "model": "Qwen/Qwen3-1.7B-Base",
            "load_in_4bit": True,
            "device_map": "auto",
            "fallback_model": "Qwen/Qwen3-1.7B-Base",
            # Memory caps for `device_map="auto"` (passed as `max_memory`). Tuned
            # for the 6 GB P40 on dsbx-host; override per machine in config.toml.
            "gpu_mem": "4500MiB",
            "cpu_mem": "13GiB",
        },
    },
    # Remote dsbx servers. Each entry is one ``dsbx serve`` instance the
    # client can connect to via ``--backend NAME`` (or by setting
    # ``[run].backend = NAME``). The bare ``remote`` name is also
    # accepted as a generic alias for the first/only entry, useful for
    # config-less smoke tests against a loopback server.
    #
    # Example:
    #   [remote.dsbx-host-py]
    #   base_url = "http://192.0.2.42:8000"
    #   timeout = 120.0
    "remote": {},
    # Web middleware (``dsbx web``). The bearer token defaults to empty so a
    # misconfigured deployment fails loudly at startup rather than serving
    # an unauthenticated API. Override via [web] in config.toml or
    # $DSBX_WEB_TOKEN / --token at the CLI.
    #
    # The nested ``[web.logging]`` table controls the upstream-request log
    # store (see decoding_sandbox/web/logging/). When ``enabled`` is true
    # every outgoing HTTP call from RemoteBackend / OpenAICompatBackend /
    # LlamaCppBackend lands as one row in the SQLite database at
    # ``db_path``, which the SvelteKit ``/logs`` tab reads back. The flush
    # task batches up to ``batch_size`` entries or flushes every
    # ``flush_interval_seconds`` seconds, whichever comes first.
    "web": {
        "api_token": "",
        "cors_origins": ["http://localhost:5173"],
        "manual_session_ttl": 3600,
        "logging": {
            "enabled": True,
            "db_path": "~/.local/share/dsbx/logs.db",
            "batch_size": 50,
            "flush_interval_seconds": 5.0,
        },
    },
    "providers": {
        "fireworks": {
            "base_url": "https://api.fireworks.ai/inference/v1",
            "api_key_env": "FIREWORKS_API_KEY",
            "default_model": "accounts/fireworks/models/gpt-oss-120b",
            "max_top_logprobs": 5,
            "supports_prompt_logprobs": True,
            "has_completions": True,
            # Fireworks supports every /v1/completions extension we care
            # about; turn them ALL on by default per the "always-on for
            # Fireworks" decision in the implementation plan. The actual
            # request body only carries each field when the caller (or
            # ``stream_native``) decides it's relevant, so wire size is
            # not affected for callers that don't ask for the feature.
            "supports_ignore_eos": True,
            "supports_perf_metrics": True,
            "supports_raw_output": True,
            "supports_sampling_mask": True,
            "supports_return_token_ids": True,
            "supports_echo_last": True,
            "supports_mirostat": True,
            "supports_typical_p_native": True,
            "supports_repetition_penalty": True,
            "supports_prompt_cache_key": True,
            "supports_service_tier": True,
            "supports_session_affinity": True,
            "supports_new_logprobs": True,
            "supports_logit_bias": True,
            "supports_combined_echo_stream": True,
            "models": [
                "accounts/fireworks/models/gpt-oss-120b",
                "accounts/fireworks/models/gpt-oss-20b",
                "accounts/fireworks/models/llama-v3p1-8b-instruct",
                "accounts/fireworks/models/qwen2p5-7b-instruct",
            ],
            # Maps each Fireworks model id to a public HuggingFace repo that
            # ships the exact ``tokenizer.json`` the Fireworks deployment
            # uses. The OpenAI-compat backend lazy-downloads that file via
            # ``hf_hub_download`` on first use and runs all local
            # tokenize/detokenize/piece calls through it. This is the
            # foundation for: (a) token-array prompt mode -- splice extra
            # ids in FRONT of the prompt so BOS-conditioning works on cloud
            # backends, (b) live token preview as the user types in the
            # Decode workbench, (c) auto-discovering bos_token_ids without
            # hard-coding them per-model. Three of these four repos are
            # public (gpt-oss-{20,120}b, Qwen2.5-7B-Instruct); the Llama
            # repo is gated and silently degrades to "no local tokenizer"
            # when ``HF_TOKEN`` is missing or lacks access -- the UI then
            # disables the prepend chip-input + live preview for that
            # model with a tooltip explaining the situation.
            "tokenizers": {
                "accounts/fireworks/models/gpt-oss-120b": "openai/gpt-oss-120b",
                "accounts/fireworks/models/gpt-oss-20b": "openai/gpt-oss-20b",
                "accounts/fireworks/models/llama-v3p1-8b-instruct": (
                    "meta-llama/Llama-3.1-8B-Instruct"
                ),
                "accounts/fireworks/models/qwen2p5-7b-instruct": (
                    "Qwen/Qwen2.5-7B-Instruct"
                ),
            },
        },
        "nim": {
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key_env": "NVIDIA_API_KEY",
            "default_model": "meta/llama-3.1-8b-instruct",
            "max_top_logprobs": 20,
            "supports_prompt_logprobs": False,
            "has_completions": False,
            "models": [
                "meta/llama-3.1-8b-instruct",
                "meta/llama-3.1-70b-instruct",
                "mistralai/mistral-7b-instruct-v0.3",
                "google/gemma-2-9b-it",
            ],
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "default_model": "meta-llama/llama-3.1-8b-instruct",
            "max_top_logprobs": 20,
            "supports_prompt_logprobs": False,
            "require_parameters": True,
            "has_completions": False,
            "models": [
                "meta-llama/llama-3.1-8b-instruct",
                "meta-llama/llama-3.1-70b-instruct",
                "qwen/qwen-2.5-7b-instruct",
                "google/gemma-2-9b-it",
            ],
        },
        "lmstudio": {
            "base_url": "http://127.0.0.1:1234/v1",
            "api_key_env": "LMSTUDIO_API_KEY",
            "default_model": "local-model",
            "max_top_logprobs": 10,
            "supports_prompt_logprobs": False,
            "has_completions": True,
            "models": ["local-model"],
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def expand(path: str | os.PathLike[str]) -> Path:
    """Expand ``~`` and environment variables in a path."""
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


@dataclass
class StorageConfig:
    hf_home: str
    pip_cache: str
    min_free_gb: float
    check_paths: list[str]


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    max_top_logprobs: int = 5
    supports_prompt_logprobs: bool = False
    require_parameters: bool = False
    # Whether the provider exposes a raw /completions endpoint (needed for our
    # samplers and whole-context echo). Chat-only providers (NIM, OpenRouter)
    # set this False and use /chat/completions for generated-token logprobs.
    has_completions: bool = False
    # Optional curated list of model names the UI should offer in its picker.
    # Always includes ``default_model``. The CLI also accepts any model name
    # via ``--model`` (or `model=` in the wire request), so this is purely
    # a UX convenience: spelling out the most useful 3-5 names per provider
    # so the browser doesn't have to know the provider's catalogue.
    models: list[str] = field(default_factory=list)
    # Per-model HuggingFace repo id that ships the matching
    # ``tokenizer.json`` -- e.g.
    # ``"accounts/fireworks/models/gpt-oss-120b": "openai/gpt-oss-120b"``.
    # ``OpenAICompatBackend`` lazy-downloads the file via
    # ``hf_hub_download`` on first use and runs every local
    # tokenize/detokenize/piece call through ``tokenizers.Tokenizer``. An
    # empty map (or a missing entry for the active model) means "no local
    # tokenizer available, fall back to the synthetic-id stub" -- the
    # backend still works for the basic text-completion paths, just with
    # ``supports_prepend_token_ids=False`` / no live token preview. Gated
    # repos (the Llama family) silently degrade to the stub when
    # ``HF_TOKEN`` is missing or lacks access; we log a warning the first
    # time and the UI surfaces a helpful tooltip.
    tokenizers: dict[str, str] = field(default_factory=dict)
    # -- provider-specific /completions extension flags ---------------------
    # These map 1:1 to optional fields in the Fireworks CompletionRequest
    # schema (https://docs.fireworks.ai/api-reference/post-completions).
    # Default ``False`` so non-Fireworks providers stay on the conservative
    # OpenAI-compatible subset; we flip each one to ``True`` for Fireworks
    # in ``_DEFAULTS`` / ``config.example.toml``. The OpenAICompatBackend
    # reads them to decide what to ship on the wire, and Capabilities
    # surfaces them so the UI can adapt (e.g. unlock the ``respect EOS``
    # checkbox once we know we can ship ``ignore_eos`` to the upstream).
    supports_ignore_eos: bool = False  # ``ignore_eos: true``
    supports_perf_metrics: bool = False  # ``perf_metrics_in_response: true``
    supports_raw_output: bool = False  # ``raw_output: true`` -> RawOutput object
    supports_sampling_mask: bool = False  # ``sampling_mask: "count"`` in NewLogProbs
    supports_return_token_ids: bool = False  # ``return_token_ids: true``
    supports_echo_last: bool = False  # ``echo_last: N`` (cheaper than ``echo: true``)
    supports_mirostat: bool = False  # ``mirostat_target`` + ``mirostat_lr``
    supports_typical_p_native: bool = False  # ``typical_p`` server-side
    supports_repetition_penalty: bool = False  # ``repetition_penalty``
    supports_prompt_cache_key: bool = False  # ``prompt_cache_key`` for KV-cache hits
    supports_service_tier: bool = False  # ``service_tier: priority`` (etc.)
    supports_session_affinity: bool = False  # ``x-session-affinity`` + R3 multi-turn id
    # NewLogProbs (``logprobs: true`` + ``top_logprobs: N``) vs legacy
    # (``logprobs: N``). The new format carries real token_id, bytes,
    # sampling_logprob, sampling_mask_count per position; legacy returns
    # ``top_logprobs[i]`` as ``{token_text: logprob}`` dicts. Fireworks
    # supports the new format; LM Studio does too. NIM/OpenRouter are
    # chat-only here so their next_distribution uses /chat/completions
    # which already speaks the new format -- this flag specifically
    # controls the /completions path. Default False keeps legacy behaviour
    # for anything we haven't explicitly verified.
    supports_new_logprobs: bool = False
    # OpenAI Completions has always accepted ``logit_bias``; we still
    # gate the UI editor on this flag so providers that quietly ignore
    # the field don't expose a knob that does nothing.
    supports_logit_bias: bool = False
    # Fireworks documents ``echo=true`` + ``stream=true`` + ``logprobs``
    # as a supported combination; when this flag is on the web layer
    # uses :meth:`OpenAICompatBackend.stream_native_with_echo` to do
    # ``include_prompt`` mode in ONE network round trip instead of two
    # (separate ``score_prompt`` + ``stream_native`` requests). Off by
    # default for safety: if a deployment doesn't tolerate the combo,
    # the user keeps the two-request fallback. Run
    # ``scripts/smoke_fireworks_echo_stream.py`` to confirm chunk order
    # before flipping this on against a non-Fireworks provider.
    supports_combined_echo_stream: bool = False

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    def known_models(self) -> list[str]:
        """``default_model`` first, followed by every distinct ``models`` entry."""
        seen: list[str] = [self.default_model]
        for m in self.models:
            if m and m not in seen:
                seen.append(m)
        return seen


@dataclass
class RemoteConfig:
    """One entry of the ``[remote.NAME]`` config table.

    Each entry maps a friendly name (``dsbx-host-py``, ``dsbx-host-hf``) to a
    running ``dsbx serve`` instance. The CLI's factory recognizes the
    name as a backend, building a ``RemoteBackend`` against ``base_url``.
    ``timeout`` is forwarded to ``httpx.Client`` -- bump it when the
    server's first request after model load can take a while.
    """

    name: str
    base_url: str
    timeout: float = 120.0


@dataclass
class Config:
    raw: dict[str, Any]
    config_path: Path | None
    secrets_env_file: str
    default_backend: str
    storage: StorageConfig
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    remotes: dict[str, RemoteConfig] = field(default_factory=dict)

    def provider(self, name: str) -> ProviderConfig:
        if name not in self.providers:
            raise KeyError(f"Unknown provider '{name}'. Known: {sorted(self.providers)}")
        return self.providers[name]

    def remote(self, name: str) -> RemoteConfig:
        if name not in self.remotes:
            raise KeyError(
                f"Unknown remote '{name}'. Known: {sorted(self.remotes) or '(none configured)'}"
            )
        return self.remotes[name]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def _find_config_file(explicit: str | os.PathLike[str] | None) -> Path | None:
    if explicit:
        p = expand(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p
    for candidate in (REPO_ROOT / "config.toml", REPO_ROOT / "config.example.toml"):
        if candidate.exists():
            return candidate
    return None


def load_env_file(path: str | os.PathLike[str]) -> int:
    """Load a simple ``KEY=value`` dotenv file into os.environ.

    Existing environment variables are NOT overwritten. Returns the number of
    new variables set. Quotes around values are stripped. Lines that are blank
    or start with ``#`` are ignored.
    """
    p = expand(path)
    if not p.exists():
        return 0
    count = 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
            count += 1
    return count


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    load_secrets: bool = True,
) -> Config:
    """Load merged configuration and (optionally) the secrets env file."""
    path = _find_config_file(config_path)
    raw = dict(_DEFAULTS)
    if path is not None:
        with path.open("rb") as fh:
            file_data = tomllib.load(fh)
        raw = _deep_merge(raw, file_data)

    secrets_env_file = raw.get("secrets_env_file", "")
    if load_secrets and secrets_env_file:
        load_env_file(secrets_env_file)

    storage = StorageConfig(
        hf_home=raw["storage"]["hf_home"],
        pip_cache=raw["storage"]["pip_cache"],
        min_free_gb=float(raw["storage"]["min_free_gb"]),
        check_paths=list(raw["storage"]["check_paths"]),
    )

    providers: dict[str, ProviderConfig] = {}
    for name, pdata in raw.get("providers", {}).items():
        providers[name] = ProviderConfig(
            name=name,
            base_url=pdata["base_url"],
            api_key_env=pdata["api_key_env"],
            default_model=pdata["default_model"],
            max_top_logprobs=int(pdata.get("max_top_logprobs", 5)),
            supports_prompt_logprobs=bool(pdata.get("supports_prompt_logprobs", False)),
            require_parameters=bool(pdata.get("require_parameters", False)),
            has_completions=bool(pdata.get("has_completions", False)),
            models=list(pdata.get("models", [])),
            tokenizers=dict(pdata.get("tokenizers", {}) or {}),
            supports_ignore_eos=bool(pdata.get("supports_ignore_eos", False)),
            supports_perf_metrics=bool(pdata.get("supports_perf_metrics", False)),
            supports_raw_output=bool(pdata.get("supports_raw_output", False)),
            supports_sampling_mask=bool(pdata.get("supports_sampling_mask", False)),
            supports_return_token_ids=bool(pdata.get("supports_return_token_ids", False)),
            supports_echo_last=bool(pdata.get("supports_echo_last", False)),
            supports_mirostat=bool(pdata.get("supports_mirostat", False)),
            supports_typical_p_native=bool(pdata.get("supports_typical_p_native", False)),
            supports_repetition_penalty=bool(pdata.get("supports_repetition_penalty", False)),
            supports_prompt_cache_key=bool(pdata.get("supports_prompt_cache_key", False)),
            supports_service_tier=bool(pdata.get("supports_service_tier", False)),
            supports_session_affinity=bool(pdata.get("supports_session_affinity", False)),
            supports_new_logprobs=bool(pdata.get("supports_new_logprobs", False)),
            supports_logit_bias=bool(pdata.get("supports_logit_bias", False)),
            supports_combined_echo_stream=bool(
                pdata.get("supports_combined_echo_stream", False)
            ),
        )

    remotes: dict[str, RemoteConfig] = {}
    for name, rdata in (raw.get("remote") or {}).items():
        if not isinstance(rdata, dict) or "base_url" not in rdata:
            raise ValueError(
                f"[remote.{name}] is missing required key 'base_url' "
                '(e.g. base_url = "http://192.0.2.42:8000").'
            )
        remotes[name] = RemoteConfig(
            name=name,
            base_url=str(rdata["base_url"]),
            timeout=float(rdata.get("timeout", 120.0)),
        )

    return Config(
        raw=raw,
        config_path=path,
        secrets_env_file=secrets_env_file,
        default_backend=raw["run"]["backend"],
        storage=storage,
        providers=providers,
        remotes=remotes,
    )
