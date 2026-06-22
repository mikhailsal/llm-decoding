"""Sensitive-field masking for log payloads.

Ported almost verbatim from ``a production proxy gateway/backend/ai_proxy/logging/masking.py``.
The two helpers are intentionally narrow so the regex used to spot a
"secret-looking" header/key is in exactly one place and the test suite can
pin it down.

We mask:

- Header keys that look secret-ish (``Authorization``, ``X-Api-Key``,
  ``proxy-authorization``...). The value is replaced with ``"abc***xyz"``
  preserving the first/last three characters so a human can still spot
  a typo from logs without leaking the secret.
- String values inside JSON request/response bodies whose KEY matches
  the same regex. We recurse into dicts and lists so nested structures
  (e.g. ``{"messages": [{"metadata": {"api_key": "..."}}]}``) get masked
  too.

This module deliberately depends on nothing beyond the stdlib so importing
it is free for tests that don't need the SQLAlchemy stack.
"""

from __future__ import annotations

import re
from typing import Any

# Match anywhere in the key name, case-insensitively. The four canonical
# names plus "secret"/"password" cover all of the credential headers/keys
# we've seen across the providers the project currently talks to. If a
# future provider names its secret differently, extend this regex --
# don't bypass it.
MASK_PATTERNS = re.compile(r"(key|token|secret|password|authorization)", re.IGNORECASE)


def mask_api_key(value: str) -> str:
    """Replace the middle of ``value`` with asterisks.

    Short strings (<= 6 chars) collapse to ``"***"`` rather than leak both
    ends; longer strings keep three chars at each side so an operator can
    eyeball "is this the production key or the dev one?" without ever
    seeing the secret middle.
    """
    if not value or len(value) <= 6:
        return "***"
    masked_len = len(value) - 6
    return f"{value[:3]}{'*' * masked_len}{value[-3:]}"


def mask_headers(headers: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a shallow copy of ``headers`` with secret-looking values masked.

    The header name match is case-insensitive (matches httpx's case-folded
    header dicts). Non-string values are passed through untouched -- httpx
    headers are always strings in practice, but defensive code is cheap.
    """
    if headers is None:
        return None
    masked: dict[str, Any] = {}
    for k, v in headers.items():
        if MASK_PATTERNS.search(str(k)):
            masked[k] = mask_api_key(str(v)) if v else v
        else:
            masked[k] = v
    return masked


def mask_sensitive_fields(data: Any) -> Any:
    """Recursively mask string values whose key matches :data:`MASK_PATTERNS`.

    Returns a freshly built structure -- the input is never mutated.
    Non-dict, non-list inputs return as-is (``None``, primitives,
    pre-masked strings...). This is fine because the only thing we
    actually want to mask is a string-valued field with a sensitive key
    name; other shapes either can't carry a credential or already passed
    through a layer that scrubbed them.
    """
    if data is None:
        return None
    if isinstance(data, list):
        return [mask_sensitive_fields(item) for item in data]
    if isinstance(data, dict):
        result: dict[str, Any] = {}
        for k, v in data.items():
            if MASK_PATTERNS.search(str(k)) and isinstance(v, str):
                result[k] = mask_api_key(v)
            else:
                result[k] = mask_sensitive_fields(v)
        return result
    return data


__all__ = ["MASK_PATTERNS", "mask_api_key", "mask_headers", "mask_sensitive_fields"]
