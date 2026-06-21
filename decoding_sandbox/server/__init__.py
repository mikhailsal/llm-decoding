"""HTTP server that exposes a long-lived in-process backend over the network.

The server is the dsbx-host-side half of the client/server split: it loads one
heavy backend (``hf`` or ``llamacpp-py``) once at startup and exposes the
``Backend`` protocol as REST + SSE so a client on another machine (the
client TUI today, the browser tomorrow) can drive it without paying the
30+ s model load on every command.

Submodules:

- ``schemas``: pydantic mirrors of the core dataclasses + request/response
  bodies. Pydantic is only needed on the server side; the client parses
  plain dicts into the existing ``core.types`` dataclasses.
- ``app``: FastAPI app factory ``make_app(backend)``. Owns the
  per-process serialization lock and the SSE generate endpoint.
- ``__main__``: ``python -m decoding_sandbox.server`` entry point (used by
  ``dsbx serve``).
"""

from __future__ import annotations

__all__ = ["make_app"]


def make_app(backend):  # pragma: no cover - thin re-export
    """Re-export so callers can ``from decoding_sandbox.server import make_app``."""
    from decoding_sandbox.server.app import make_app as _make

    return _make(backend)
