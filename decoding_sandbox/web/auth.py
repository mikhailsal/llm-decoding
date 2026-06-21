"""Bearer-token authentication for the dsbx web middleware.

The middleware uses a *single* shared bearer token (the "single-user" model
chosen during planning). The token is supplied via:

1. ``--token`` on the ``dsbx web`` CLI, or
2. ``$DSBX_WEB_TOKEN``, or
3. ``[web].api_token`` in ``config.toml``.

At app construction time the token string is captured once and stored in a
closure; every request through the :func:`require_bearer` dependency compares
the supplied ``Authorization: Bearer <token>`` against it in constant time
(``hmac.compare_digest``). The dependency raises HTTP 401 on any mismatch,
missing header, or wrong scheme.

The unauthenticated ``/api/v1/health`` endpoint is the only exception -- it is
deliberately public so a reverse proxy can probe liveness without ever seeing
the token. Every other ``/api/v1/*`` route MUST attach the dependency.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import Header, HTTPException, status


@dataclass(frozen=True)
class AuthConfig:
    """Captured by the app factory and held in a closure for the dependency.

    The token is required to be non-empty at construction. The ``server_label``
    is purely advisory -- it shows up on the ``/api/v1/health`` ping so an
    operator can tell different middleware instances apart in a screenshot.
    """

    token: str
    server_label: str = "dsbx-web"

    def __post_init__(self) -> None:
        # Defensive: an empty token would make every request authenticate,
        # which is the opposite of what we want. The CLI surfaces a clearer
        # error than a generic empty-string raise -- this check is the last
        # line of defense if someone constructs ``AuthConfig`` programmatically.
        if not self.token or not self.token.strip():
            raise ValueError(
                "AuthConfig.token must be a non-empty string. Set it via "
                "--token, $DSBX_WEB_TOKEN, or [web].api_token in config.toml."
            )


def _unauthorized(detail: str) -> HTTPException:
    # Always render the same WWW-Authenticate header so a browser knows to
    # prompt for credentials. The body's ``detail`` field varies by reason
    # (no header vs. wrong scheme vs. wrong token), which is convenient for
    # debugging but never reveals a configured token.
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": 'Bearer realm="dsbx-web"'},
    )


def make_require_bearer(
    cfg: AuthConfig,
) -> Callable[..., Awaitable[None]] | Callable[..., None]:
    """Build the FastAPI dependency that gates every authenticated route.

    The returned callable is async-compatible (FastAPI awaits dependency
    return values even when they're sync), accepts an optional
    ``Authorization`` header, and either returns ``None`` (success) or
    raises ``HTTPException(401)``. ``hmac.compare_digest`` is used so a
    timing oracle cannot leak the configured token byte-by-byte.
    """

    def require_bearer(authorization: str | None = Header(default=None)) -> None:
        if authorization is None:
            raise _unauthorized("missing Authorization header")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer":
            raise _unauthorized("Authorization scheme must be 'Bearer'")
        # Compare the full provided token against the configured one in
        # constant time. ``compare_digest`` requires both arguments to be
        # the same type; we ensure ``str`` on both sides so the comparison
        # is well-defined for non-ASCII tokens (Python 3 strings).
        if not hmac.compare_digest(token, cfg.token):
            raise _unauthorized("invalid bearer token")
        # Successful auth returns ``None`` (FastAPI dependencies that return
        # ``None`` simply gate access without injecting a value).
        return None

    return require_bearer


__all__ = ["AuthConfig", "make_require_bearer"]
