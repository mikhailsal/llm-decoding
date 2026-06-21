"""dsbx web -- authenticated middleware between browser and decoding backends.

The web package is a thin FastAPI layer sitting between a browser UI and the
existing ``Backend`` ecosystem (``RemoteBackend`` against ``dsbx serve`` on
``dsbx-host``, ``OpenAICompatBackend`` against cloud providers, in-process engines
where applicable). Its single non-obvious job is to *absorb every secret*:
provider API keys, the LAN address of ``dsbx-host``, the contents of
``secrets_env_file`` -- none of it ever crosses the wire to the browser.

The browser sees:

- An opaque list of backend names (``dsbx-host-py``, ``fireworks``, ...) plus their
  ``Capabilities`` flags.
- A bearer-token-protected REST + SSE API for inspect/generate/manual/spec.
- Friendly error messages without internal addresses or stack traces.

The browser never sees:

- ``base_url`` for remote dsbx servers.
- ``api_key_env`` or any actual API key.
- The path to ``secrets_env_file``.
- Wind's LAN IP, in any field.

See :func:`decoding_sandbox.web.app.make_web_app` for the FastAPI entry point.
"""

from __future__ import annotations

from decoding_sandbox.web.app import make_web_app

__all__ = ["make_web_app"]
