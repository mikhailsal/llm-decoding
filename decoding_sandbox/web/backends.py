"""Lazy, cache-on-first-use registry of named Backends for the web middleware.

The middleware is single-process by design (the existing dsbx_serve already
serves the heavy work; this layer is glue + auth + secrets-hiding). We hold
one ``Backend`` instance per logical name and reuse it for every browser
request. A ``threading.Lock`` per backend serializes calls into that instance,
mirroring the locking policy that ``decoding_sandbox/server/app.py`` already
uses for the same KV-cache-correctness reason.

What we deliberately do NOT cache:

- Speculative-decoding HF target/draft pairs. Those are heavy and pin VRAM;
  ``/api/v1/spec/stream`` builds them on demand and closes them when the
  stream ends (mirrors :func:`decoding_sandbox.cli.app.cmd_spec`).

Public listing rules:

- For ``[remote.NAME]`` entries we expose ``{name, family="remote"}`` and never
  the ``base_url`` -- even in error messages. (The browser receives "remote
  backend error" with a generic explanation; the operator can read the real
  cause from the middleware's stderr log.)
- For ``[providers.NAME]`` entries we expose ``{name, family="cloud"}``; the
  ``api_key_env`` is read by the middleware itself when it constructs the
  ``OpenAICompatBackend`` -- the browser never sees it.
- ``local`` backends (``hf`` / ``llamacpp`` / ``llamacpp-py``) are listed too,
  but heavy-load is deferred to first use.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Literal

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config
from decoding_sandbox.core.factory import LOCAL_BACKENDS, build_backend
from decoding_sandbox.web.schemas import BackendInfo

# Default TTL for the per-backend model-catalogue cache. Cloud providers
# don't churn their catalogues on minute scales but they do retire models
# every few weeks (Fireworks notably so), so a 6h refresh is a reasonable
# balance between "fresh enough to notice a new GPT-OSS variant" and "not
# hammering /v1/accounts/fireworks/models on every page load".
MODEL_LIST_TTL_S = 6 * 3600.0

ModelListSource = Literal["live", "cached", "static", "fallback"]


@dataclass
class ModelListEntry:
    """Cached result of one ``list_models`` lookup for one backend name.

    ``source`` records how we got the list so the UI can show "served from
    cache (3h ago)" vs "fresh from provider" vs "fell back to static list
    because the live fetch failed". ``note`` is a short human-readable
    string for the same purpose -- safe to surface to the browser because
    it contains no URL or env-var name.
    """

    models: list[str]
    source: ModelListSource
    fetched_at: float
    note: str = ""


log = logging.getLogger("decoding_sandbox.web.backends")


@dataclass
class _BackendEntry:
    """A logical backend slot. The instance is built lazily on first use.

    ``family`` mirrors :class:`BackendInfo.family`. ``unavailable_reason`` is
    set when a list-time check (e.g. missing API key for a cloud provider)
    can predict that the backend can't be used; the UI then shows it as
    disabled with that reason in a tooltip.

    ``instance`` is the default-model instance (used for non-cloud families
    and as the fallback when a cloud caller doesn't specify a model).
    ``cloud_variants`` is the per-(model) cache used only by cloud providers
    where the user can pick a different model per request -- those
    backends are cheap to construct (one ``httpx.Client``) so a tiny LRU
    of recently-seen names beats reloading on every request.

    The single ``lock`` covers *all* variants of the same logical backend
    name; that's intentional. Two concurrent generates on the same cloud
    provider with two different models would still hit the same upstream
    rate-limit bucket, so serializing them keeps the load predictable.
    """

    name: str
    family: str  # "remote" | "cloud" | "local"
    instance: Backend | None = None
    cloud_variants: dict[str, Backend] = None  # type: ignore[assignment]
    lock: threading.Lock = None  # type: ignore[assignment]
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        if self.lock is None:
            self.lock = threading.Lock()
        if self.cloud_variants is None:
            self.cloud_variants = {}


class BackendRegistry:
    """Owns every backend instance the middleware ever talks to.

    Use as a context manager so all loaded instances are closed cleanly on
    shutdown (releases httpx clients for remote backends, VRAM for HF, etc.).

    ``logging_enabled`` controls whether outgoing HTTP calls from
    ``RemoteBackend`` / ``OpenAICompatBackend`` / ``LlamaCppBackend`` are
    captured into the upstream-request log. When True the registry
    builds a fresh :class:`LoggingTransport` per backend (tagged with
    the backend name / family / provider) and threads it through
    :func:`build_backend`. ``loop`` is the asyncio event loop those
    transports should schedule their enqueues on; captured here because
    the actual httpx call site runs on a worker thread.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        logging_enabled: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._cfg = cfg
        self._entries: dict[str, _BackendEntry] = {}
        self._models_cache: dict[str, ModelListEntry] = {}
        # Independent lock for the model-list cache so a long-running
        # generate stream (holding a per-backend lock) doesn't block a
        # browser asking "what models are available?".
        self._models_lock = threading.Lock()
        self._logging_enabled = bool(logging_enabled)
        self._loop = loop
        self._enumerate()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the asyncio loop after construction.

        ``make_web_app`` builds the registry synchronously (before the
        FastAPI lifespan fires), so the actual event loop reference
        becomes available only later. Calling this from the startup
        hook lets every transport built afterwards know where to
        schedule enqueues. Transports built BEFORE the attach receive
        ``loop=None`` and fall back to a private ``asyncio.run`` -- only
        the registry constructor cares about this distinction.
        """
        self._loop = loop

    def _build_transport(self, entry: _BackendEntry):
        """Build a fresh :class:`LoggingTransport` tagged for this backend.

        Returns ``None`` when logging is disabled so the caller can pass
        the result straight to ``build_backend(..., transport=...)``
        without a branch. The transport carries the registered backend
        name, family, provider name (when cloud), and the upstream base
        URL so the log row's UI shows them without an extra join.
        """
        if not self._logging_enabled:
            return None
        # Local import: keeping the import lazy lets the CLI keep working
        # without SQLAlchemy installed (the ``web`` extra brings it in).
        from decoding_sandbox.web.logging.transport import LoggingTransport

        base_url = ""
        if entry.family == "remote":
            rc = self._cfg.remotes.get(entry.name)
            base_url = rc.base_url if rc is not None else ""
        elif entry.family == "cloud":
            prov = self._cfg.providers.get(entry.name)
            base_url = prov.base_url if prov is not None else ""
        elif entry.name == "llamacpp":
            base_url = self._cfg.get("local", "llamacpp", "base_url", default="") or ""

        return LoggingTransport(
            backend_name=entry.name,
            backend_family=entry.family,
            provider_name=entry.name if entry.family == "cloud" else None,
            upstream_base_url=base_url,
            loop=self._loop,
        )

    # ------------------------------------------------------- discovery
    def _enumerate(self) -> None:
        """Build the *logical* listing without instantiating anything heavy.

        This is what ``/api/v1/info`` calls. Heavy work (loading a GGUF,
        running a forward pass to read EOS ids) only happens when the
        browser actually picks a backend.
        """
        # Remote dsbx servers come first because that's the typical hot path.
        for name in sorted(self._cfg.remotes):
            self._entries[name] = _BackendEntry(name=name, family="remote")

        # Cloud providers. Missing API key is the only check we can do at
        # listing time -- we expose the provider but mark it unavailable so
        # the UI dims the option and shows why.
        for name in sorted(self._cfg.providers):
            prov = self._cfg.providers[name]
            unavail = ""
            if not prov.api_key() and name != "lmstudio":
                unavail = f"missing {prov.api_key_env}"
            self._entries[name] = _BackendEntry(
                name=name, family="cloud", unavailable_reason=unavail
            )

        # Local in-process engines. They are always *listed*; whether they
        # can actually load on this host is discovered only when the first
        # request arrives.
        for name in LOCAL_BACKENDS:
            self._entries[name] = _BackendEntry(name=name, family="local")

    # --------------------------------------------------- public listing
    def list_public(self) -> list[BackendInfo]:
        """Listing payload for ``GET /api/v1/info``.

        Capabilities are read from already-loaded backends if available;
        otherwise we leave the field null so the UI knows to ask later.
        Importantly, NO field in the returned payload contains a URL or a
        secret -- :func:`tests.test_web_info` enforces this.
        """
        from decoding_sandbox.server.schemas import (
            WireCapabilities,
            capabilities_to_wire,
        )

        out: list[BackendInfo] = []
        for entry in self._entries.values():
            caps = None
            if entry.instance is not None:
                try:
                    caps = capabilities_to_wire(entry.instance.capabilities)
                except Exception:  # noqa: BLE001
                    caps = None
            # For cloud providers we can synthesize the capability envelope
            # from static ProviderConfig data even before a single request
            # has loaded the backend. Without this, the UI couldn't enforce
            # provider-specific limits (Fireworks' 5-logprob ceiling, etc.)
            # until *after* the first generate call, which was the exact
            # "silently ignored ``top_k`` field" UX the audit flagged.
            # The synthetic version omits a stable ``eos_token_ids`` list
            # because that depends on the tokenizer the upstream uses, but
            # the fields the UI cares about up front (max_top_logprobs,
            # prompt_logprobs, full_vocab) come straight from config.
            if caps is None and entry.family == "cloud":
                prov = self._cfg.providers.get(entry.name)
                if prov is not None:
                    # Mirror EVERY supports_* flag we publish on the
                    # loaded backend's capabilities here too -- otherwise
                    # the UI sees a "stripped down" capability set for
                    # cloud backends until the first real request lands
                    # and the proper OpenAICompatBackend instance is
                    # constructed. That was the exact "respect EOS
                    # locked for Fireworks before first run" UX glitch
                    # the manual Chrome MCP check caught.
                    #
                    # ``generation_disabled`` falls out of
                    # ``has_completions=false`` (chat-only providers --
                    # NIM, OpenRouter). The frontend backend picker
                    # uses the flag to render the option as
                    # ``<option disabled>`` with ``notes`` as tooltip
                    # text, and the generate-stream route enforces the
                    # same gate authoritatively. We pre-fill ``notes``
                    # with the same wording the loaded backend would
                    # publish so the tooltip is identical pre- and
                    # post-first-use.
                    is_chat_only = not bool(prov.has_completions)
                    if is_chat_only:
                        notes = (
                            "chat-only provider; generation disabled "
                            "until proper chat-mode UI lands"
                        )
                    else:
                        notes = (
                            "static caps from provider config (backend not yet loaded)"
                        )
                    caps = WireCapabilities(
                        name=f"openai_compat:{entry.name}",
                        full_vocab=False,
                        prompt_logprobs=bool(prov.supports_prompt_logprobs),
                        max_top_logprobs=int(prov.max_top_logprobs),
                        can_force_token=bool(prov.has_completions),
                        notes=notes,
                        eos_token_ids=[],
                        # Cloud providers tokenize ``prompt: str``
                        # server-side; we can't safely splice extra
                        # token ids in front, so we report no BOS and
                        # gate the UI's prepend chip-input off.
                        bos_token_ids=[],
                        supports_ignore_eos=bool(prov.supports_ignore_eos),
                        supports_perf_metrics=bool(prov.supports_perf_metrics),
                        supports_service_tier=bool(prov.supports_service_tier),
                        supports_sampling_mask=bool(prov.supports_sampling_mask),
                        supports_raw_output=bool(prov.supports_raw_output),
                        supports_logit_bias=bool(prov.supports_logit_bias),
                        supports_combined_echo_stream=bool(
                            prov.supports_combined_echo_stream
                        ),
                        supports_prepend_token_ids=False,
                        generation_disabled=is_chat_only,
                    )
            label = self._public_label(entry)
            loaded_model, suggested, editable = self._public_model_info(entry)
            out.append(
                BackendInfo(
                    name=entry.name,
                    label=label,
                    family=entry.family,  # type: ignore[arg-type]
                    capabilities=caps,
                    available=not entry.unavailable_reason,
                    note=entry.unavailable_reason,
                    loaded_model=loaded_model,
                    suggested_models=suggested,
                    model_editable=editable,
                )
            )
        return out

    def _public_model_info(self, entry: _BackendEntry) -> tuple[str | None, list[str], bool]:
        """Compute (loaded_model, suggested_models, model_editable) for a row.

        We compute this from *static* config wherever possible so the
        listing endpoint stays cheap (no network calls). For an
        already-loaded ``remote`` backend we DO have a cached
        ``loaded_model`` string from the upstream ``/v1/info`` response
        that came back at first construction, so we surface that too.
        """
        if entry.family == "cloud":
            prov = self._cfg.providers.get(entry.name)
            if prov is None:
                return None, [], False
            return prov.default_model, prov.known_models(), True
        if entry.family == "remote":
            # If the remote is already loaded we know exactly what's running;
            # otherwise we don't know (the upstream hasn't been pinged yet)
            # and leave it null -- the browser shows "unknown until loaded".
            loaded = None
            if entry.instance is not None:
                loaded = getattr(entry.instance, "loaded_model", None)
            return loaded, ([loaded] if loaded else []), False
        # family == "local"
        if entry.name == "hf":
            hf_cfg = self._cfg.get("local", "hf", default={})
            model = hf_cfg.get("model")
            return model, ([model] if model else []), False
        if entry.name == "llamacpp":
            lc_cfg = self._cfg.get("local", "llamacpp", default={})
            model = lc_cfg.get("model")
            return model, ([model] if model else []), False
        if entry.name == "llamacpp-py":
            lp_cfg = self._cfg.get("local", "llamacpp_py", default={})
            glob = lp_cfg.get("model_glob") or ""
            model = lp_cfg.get("model_path") or glob.replace("**/", "")
            return model, ([model] if model else []), False
        return None, [], False

    # --------------------------------------------------- model catalogue
    def list_models(self, name: str, *, refresh: bool = False) -> ModelListEntry:
        """Return the model catalogue for ``name``, hitting the wire at most once per TTL.

        Behaviour by family:

        - ``cloud`` providers: ask the upstream's catalogue endpoint
          (OpenAI-compat ``/models`` for NIM / OpenRouter / LM Studio, the
          per-account Fireworks endpoint for that one). The returned list
          is UNIONED with the curated ``[providers.NAME].models`` so a
          locally-pinned name stays visible even when the provider has
          retired it from the catalogue. Failures fall back to the
          curated list and ``source="fallback"``; the ``note`` is a
          short error category (no URLs).
        - ``remote`` / ``local`` backends: no network call. Return the
          single-element list that ``list_public`` already exposes via
          ``suggested_models``; ``source="static"``.

        ``refresh=True`` invalidates the cache for this name and forces a
        fresh fetch even if the cached entry is still warm.
        """
        entry = self.get(name)
        cached = None
        now = time.time()
        with self._models_lock:
            cached = self._models_cache.get(name)
            if cached and not refresh and (now - cached.fetched_at) < MODEL_LIST_TTL_S:
                # Hand back a copy so callers can mutate freely.
                return ModelListEntry(
                    models=list(cached.models),
                    source="cached",
                    fetched_at=cached.fetched_at,
                    note=cached.note,
                )

        if entry.family != "cloud":
            loaded_model, suggested, _ = self._public_model_info(entry)
            result = ModelListEntry(
                models=list(suggested),
                source="static",
                fetched_at=now,
                note=(
                    f"{entry.family} backends don't expose a catalogue; "
                    "showing the configured model only"
                ),
            )
            with self._models_lock:
                self._models_cache[name] = result
            return ModelListEntry(
                models=list(result.models),
                source=result.source,
                fetched_at=result.fetched_at,
                note=result.note,
            )

        # Cloud path: try a live fetch via the OpenAICompatBackend's helper.
        prov = self._cfg.providers.get(name)
        if prov is None:
            raise LookupError(f"backend {name!r} is not a configured provider")
        curated = list(prov.known_models())
        if entry.unavailable_reason:
            # No API key set -- we can't actually list. The curated list is
            # the best the browser can show.
            result = ModelListEntry(
                models=curated,
                source="fallback",
                fetched_at=now,
                note=f"cannot fetch live catalogue ({entry.unavailable_reason})",
            )
            with self._models_lock:
                self._models_cache[name] = result
            return ModelListEntry(
                models=list(result.models),
                source=result.source,
                fetched_at=result.fetched_at,
                note=result.note,
            )
        try:
            from decoding_sandbox.backends.openai_compat import OpenAICompatBackend

            # We deliberately spin up a fresh OpenAICompatBackend instead of
            # reusing entry.instance / one of entry.cloud_variants: the
            # catalogue endpoint is per-provider, not per-model, so the
            # cached chat client is irrelevant here. Construction is cheap
            # (one httpx.Client) and we close it immediately. The probe
            # also gets a logging transport so catalogue fetches end up
            # in the same upstream-request log as chat/completions calls.
            probe = OpenAICompatBackend(prov, transport=self._build_transport(entry))
            try:
                live = probe.fetch_available_models()
            finally:
                probe.close()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "dsbx-web: failed to fetch %r model catalogue: %s", name, exc.__class__.__name__
            )
            result = ModelListEntry(
                models=curated,
                source="fallback",
                fetched_at=now,
                note=(
                    f"live catalogue unavailable ({exc.__class__.__name__}); "
                    "showing curated suggestions"
                ),
            )
            with self._models_lock:
                self._models_cache[name] = result
            return ModelListEntry(
                models=list(result.models),
                source=result.source,
                fetched_at=result.fetched_at,
                note=result.note,
            )

        # Union live ∪ curated, preserving curated order at the front so
        # the user's favourites stay near the top.
        seen: set[str] = set()
        merged: list[str] = []
        for m in curated:
            if m and m not in seen:
                seen.add(m)
                merged.append(m)
        for m in live:
            if m and m not in seen:
                seen.add(m)
                merged.append(m)
        result = ModelListEntry(
            models=merged,
            source="live",
            fetched_at=now,
            note=f"{len(live)} from provider, {len(curated)} curated",
        )
        with self._models_lock:
            self._models_cache[name] = result
        return ModelListEntry(
            models=list(result.models),
            source=result.source,
            fetched_at=result.fetched_at,
            note=result.note,
        )

    def invalidate_models_cache(self, name: str | None = None) -> None:
        """Drop the cached catalogue for one backend (or all if ``name`` is None)."""
        with self._models_lock:
            if name is None:
                self._models_cache.clear()
            else:
                self._models_cache.pop(name, None)

    @staticmethod
    def _public_label(entry: _BackendEntry) -> str:
        """Human-friendly label that explicitly hides any address/URL.

        Crucially, the label is derived purely from the logical name -- not
        the ``base_url`` -- so renaming a ``[remote.NAME]`` block doesn't
        accidentally surface its hostname in the UI.
        """
        if entry.family == "remote":
            return f"{entry.name} (remote dsbx server)"
        if entry.family == "cloud":
            return f"{entry.name} (cloud provider)"
        return f"{entry.name} (local engine)"

    # -------------------------------------------------------- get / use
    def names(self) -> list[str]:
        return list(self._entries)

    def get(self, name: str) -> _BackendEntry:
        if name not in self._entries:
            raise KeyError(f"unknown backend {name!r}")
        return self._entries[name]

    def ensure_loaded(self, name: str, model: str | None = None) -> Backend:
        """Load the backend on first use; reuse on every subsequent call.

        For non-cloud families ``model`` is ignored (changing the model in
        an HF / llamacpp-py / remote backend means reloading or restarting
        a remote process, which the middleware deliberately does NOT do
        silently). For cloud providers a fresh ``OpenAICompatBackend`` is
        cached per distinct ``model`` string; absent ``model`` defaults to
        the provider's ``default_model``.

        Heavy local engines (HF, llamacpp-py) print a load banner to stderr
        but never to the response body so the browser stays oblivious.
        """
        entry = self.get(name)
        if entry.unavailable_reason:
            # Surface the *category* of the problem (missing key) but
            # not the value: "missing FIREWORKS_API_KEY" is fine; an
            # actual key would obviously not be.
            raise LookupError(f"backend {name!r} is unavailable: {entry.unavailable_reason}")

        if entry.family == "cloud" and model:
            # Reuse if we've already built this model variant.
            cached = entry.cloud_variants.get(model)
            if cached is not None:
                return cached
            log.info(
                "dsbx-web: building cloud backend %r with model %r on first use",
                name,
                model,
            )
            inst = build_backend(
                name, self._cfg, model=model, transport=self._build_transport(entry)
            )
            entry.cloud_variants[model] = inst
            return inst

        if entry.instance is None:
            log.info("dsbx-web: building backend %r on first use", name)
            entry.instance = build_backend(
                name, self._cfg, model=None, transport=self._build_transport(entry)
            )
        return entry.instance

    def use(self, name: str, model: str | None = None) -> "_LockedBackend":
        """Context manager that yields the live backend under its lock.

        Mirrors the lock usage in ``server/app.py``. The lock is held only
        for the duration of the ``with`` block; SSE generation uses it for
        the entire stream so a parallel ``inspect`` can't corrupt the KV
        cache mid-decode. The lock is per *logical* backend name so two
        concurrent cloud calls with different models still serialize -- we
        want the upstream rate-limit bucket to stay predictable.
        """
        entry = self.get(name)
        backend = self.ensure_loaded(name, model)
        return _LockedBackend(backend, entry.lock)

    # ---------------------------------------------------------- lifecycle
    def close_all(self) -> None:
        """Close every backend we ever loaded.

        Called from the FastAPI shutdown hook and from the registry's
        ``__exit__``. Each ``close`` is run in a try/except so a bad close
        doesn't shadow the rest.
        """
        for entry in self._entries.values():
            if entry.instance is not None:
                try:
                    entry.instance.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning("dsbx-web: error closing backend %r: %s", entry.name, exc)
                finally:
                    entry.instance = None
            for mkey, inst in list(entry.cloud_variants.items()):
                try:
                    inst.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "dsbx-web: error closing cloud variant %r/%r: %s",
                        entry.name,
                        mkey,
                        exc,
                    )
            entry.cloud_variants.clear()

    def __enter__(self) -> "BackendRegistry":
        return self

    def __exit__(self, *_excinfo: object) -> None:
        self.close_all()


class _LockedBackend:
    """Tiny holder for ``backend, lock`` so call sites can ``with`` cleanly."""

    def __init__(self, backend: Backend, lock: threading.Lock) -> None:
        self._backend = backend
        self._lock = lock
        self._held = False

    def __enter__(self) -> Backend:
        self._lock.acquire()
        self._held = True
        return self._backend

    def __exit__(self, *_excinfo: object) -> None:
        if self._held:
            self._lock.release()
            self._held = False


def iter_backend_names(cfg: Config) -> Iterator[str]:
    """Public enumeration helper used by tests to assert listing parity."""
    yield from sorted(cfg.remotes)
    yield from sorted(cfg.providers)
    yield from LOCAL_BACKENDS


__all__ = [
    "BackendRegistry",
    "ModelListEntry",
    "MODEL_LIST_TTL_S",
    "iter_backend_names",
]
