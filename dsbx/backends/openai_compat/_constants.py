"""Shared module-level constants for the OpenAI-compatible backend package."""

from __future__ import annotations

# HTTP statuses we treat as worth one more try. 429 is the headline case
# (per-account RPS throttle on serverless cloud endpoints); the transient
# 5xx group catches "the gateway in front of the model server hiccuped".
# 400/401/403/404 are NOT retried -- they signal a request-shape or auth
# problem that won't fix itself.
_RETRIABLE_STATUSES = frozenset({429, 502, 503, 504})

# Number of retries on top of the initial attempt. 3 means the caller waits
# at most ~7s (1+2+4) of pure backoff before the call fails, which keeps the
# UX snappy while absorbing a brief burst hitting the upstream limit.
_MAX_RETRIES = 3

# Base delay (seconds) for the exponential-backoff fallback when the server
# did not send a Retry-After header. With _MAX_RETRIES=3 the schedule is
# roughly 1s, 2s, 4s, each with up to 250 ms of jitter to desynchronize
# parallel callers (relevant when several browser tabs hit the same backend).
_BASE_BACKOFF_S = 1.0

# Universally mappable samplers (every OpenAI-compat /completions
# endpoint understands these). For ``typical`` and ``mirostat`` we
# additionally check ProviderConfig.supports_typical_p_native /
# supports_mirostat at request time -- those are Fireworks-extensions
# and we don't want to silently ship dead params to providers that
# would either 400 the request or accept-and-ignore the field.
# ``custom`` obviously can't run remotely.
_NATIVE_SAMPLERS: frozenset[str] = frozenset({"greedy", "temperature", "top_k", "top_p", "min_p"})
