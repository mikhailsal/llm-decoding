"""OpenAI-compatible cloud/local backend (Fireworks / NIM / OpenRouter / LM Studio).

Cloud providers tokenize server-side and only return logprobs as *strings*, so
this backend works in text space: it assigns a stable synthetic integer id to
each distinct token string it sees (so the rest of the system, which speaks
token ids, keeps working unchanged).

Capabilities differ by provider, surfaced via ``Capabilities`` so the UI adapts:
- Fireworks: /completions with logprobs (our samplers) AND whole-context via
  ``echo`` (the only cloud path to per-prompt-token logprobs).
- NIM / OpenRouter: chat-only -> generated-token logprobs via /chat/completions
  (no raw continuation, no whole-context echo).
- LM Studio: local OpenAI server with /completions.

The implementation is split across this package for readability:
- :mod:`._http` -- retry/backoff and the request chokepoint.
- :mod:`._tokenizer` -- lazy local HF tokenizer + surface-text helpers.
- :mod:`._parsing` -- token-id interning and response candidate parsing.
- :mod:`._fireworks_ext` -- Fireworks-extension request shaping + catalogue.
- :mod:`._streaming` / :mod:`._streaming_echo` -- the SSE decode loops.
- :mod:`.backend` -- the :class:`OpenAICompatBackend` class itself.

``httpx`` and ``random`` are re-exported here so existing tests can monkeypatch
``openai_compat.httpx`` / ``openai_compat.random`` against the shared modules.
"""

from __future__ import annotations

import random as random

import httpx as httpx

from dsbx.backends.openai_compat._constants import (
    _BASE_BACKOFF_S as _BASE_BACKOFF_S,
)
from dsbx.backends.openai_compat._constants import (
    _MAX_RETRIES as _MAX_RETRIES,
)
from dsbx.backends.openai_compat._constants import (
    _RETRIABLE_STATUSES as _RETRIABLE_STATUSES,
)
from dsbx.backends.openai_compat._http import (
    _parse_retry_after as _parse_retry_after,
)
from dsbx.backends.openai_compat.backend import (
    OpenAICompatBackend as OpenAICompatBackend,
)

__all__ = ["OpenAICompatBackend"]
