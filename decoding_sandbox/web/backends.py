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

import logging
import threading
from dataclasses import dataclass
from typing import Iterator

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config
from decoding_sandbox.core.factory import LOCAL_BACKENDS, build_backend
from decoding_sandbox.web.schemas import BackendInfo

log = logging.getLogger("decoding_sandbox.web.backends")


@dataclass
class _BackendEntry:
    """A logical backend slot. The instance is built lazily on first use.

    ``family`` mirrors :class:`BackendInfo.family`. ``unavailable_reason`` is
    set when a list-time check (e.g. missing API key for a cloud provider)
    can predict that the backend can't be used; the UI then shows it as
    disabled with that reason in a tooltip.
    """

    name: str
    family: str  # "remote" | "cloud" | "local"
    instance: Backend | None = None
    lock: threading.Lock = None  # type: ignore[assignment]
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        if self.lock is None:
            self.lock = threading.Lock()


class BackendRegistry:
    """Owns every backend instance the middleware ever talks to.

    Use as a context manager so all loaded instances are closed cleanly on
    shutdown (releases httpx clients for remote backends, VRAM for HF, etc.).
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._entries: dict[str, _BackendEntry] = {}
        self._enumerate()

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
        from decoding_sandbox.server.schemas import capabilities_to_wire

        out: list[BackendInfo] = []
        for entry in self._entries.values():
            caps = None
            if entry.instance is not None:
                try:
                    caps = capabilities_to_wire(entry.instance.capabilities)
                except Exception:  # noqa: BLE001
                    caps = None
            label = self._public_label(entry)
            out.append(
                BackendInfo(
                    name=entry.name,
                    label=label,
                    family=entry.family,  # type: ignore[arg-type]
                    capabilities=caps,
                    available=not entry.unavailable_reason,
                    note=entry.unavailable_reason,
                )
            )
        return out

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

    def ensure_loaded(self, name: str) -> Backend:
        """Load the backend on first use; reuse on every subsequent call.

        Heavy local engines (HF, llamacpp-py) print a load banner to stderr
        but never to the response body so the browser stays oblivious.
        """
        entry = self.get(name)
        if entry.instance is None:
            if entry.unavailable_reason:
                # Surface the *category* of the problem (missing key) but
                # not the value: "missing FIREWORKS_API_KEY" is fine; an
                # actual key would obviously not be.
                raise LookupError(f"backend {name!r} is unavailable: {entry.unavailable_reason}")
            log.info("dsbx-web: building backend %r on first use", name)
            entry.instance = build_backend(name, self._cfg, model=None)
        return entry.instance

    def use(self, name: str) -> "_LockedBackend":
        """Context manager that yields the live backend under its lock.

        Mirrors the lock usage in ``server/app.py``. The lock is held only
        for the duration of the ``with`` block; SSE generation uses it for
        the entire stream so a parallel ``inspect`` can't corrupt the KV
        cache mid-decode.
        """
        entry = self.get(name)
        backend = self.ensure_loaded(name)
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


__all__ = ["BackendRegistry", "iter_backend_names"]
