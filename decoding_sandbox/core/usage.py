"""Per-run usage accounting (HTTP requests, prompt/completion tokens, notes).

A run's "usage" is a small mutable dict; the caller initializes it, hands
it to a backend that opts into tracking, and reads the populated dict
back when the call completes. Backends that don't implement the
:class:`UsageAware` protocol below simply leave the sink untouched and
the caller fills in what it can locally (e.g. ``completion_tokens`` from
the number of emitted ``GenStep``s).

Why not :mod:`contextvars`? It looked attractive at first but starlette
iterates sync ``StreamingResponse`` bodies via :func:`iterate_in_threadpool`,
which dispatches each ``__next__`` to a fresh worker thread with the
calling task's context COPIED in (mutations don't propagate back, and the
reset token tied to one worker's context won't validate against another's).
A plain instance attribute on the backend, serialized by the per-backend
``threading.Lock`` we already hold for the duration of the stream, is the
simplest correct primitive here.

Concurrency note: the :mod:`decoding_sandbox.web.backends` registry holds
a per-backend ``threading.Lock`` for the entire duration of a generate or
inspect stream (see ``_use_backend``). That serializes two concurrent
callers against the same backend instance, so backends can store an
"active sink" as a single instance attribute without ever stepping on
each other's accounting.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Plain dict-of-Any so callers can ``json.dumps`` it straight onto the
# SSE wire without an adapter. The well-known keys are listed here as
# documentation but the type stays open so backends can ferry extras
# (e.g. provider-specific "cached_tokens" some day).
#
# - requests:          int counter; each HTTP attempt against an upstream
#                      provider increments this (retries included on purpose
#                      -- we want users to see RPS pressure, not just
#                      successful calls).
# - prompt_tokens:     authoritative count from provider's usage block when
#                      available; falls back to local ``len(tokenize)`` for
#                      backends whose tokenizer is reliable.
# - completion_tokens: same idea, provider's count preferred; falls back
#                      to the number of emitted ``GenStep``s.
# - total_tokens:      provider's sum when given; otherwise computed by the
#                      caller as prompt + completion.
# - notes:             list[str] of human-readable advisories surfaced by
#                      the backend (e.g. "respect_eos=False is unsupported
#                      by this cloud provider").
UsageSink = dict[str, Any]


@runtime_checkable
class UsageAware(Protocol):
    """Backends that can populate a :class:`UsageSink` for the current call.

    A caller (typically :mod:`decoding_sandbox.web.streaming`) sets the
    active sink on the backend BEFORE invoking a method, then reads the
    populated dict back AFTER the call returns. Implementations are
    expected to write counters via the helpers in this module so the
    field names stay consistent across backends.

    The contract is single-call: the caller MUST clear the sink (with
    ``set_active_usage(None)``) once it's done; otherwise a later call
    on the same backend would mutate a stale dict. The per-backend lock
    held by :mod:`decoding_sandbox.web.backends` makes that easy --
    set, run, clear, all within the same ``with`` block.
    """

    def set_active_usage(self, sink: UsageSink | None) -> None: ...


def make_sink() -> UsageSink:
    """Construct a fresh sink with all known fields initialised.

    Centralised so every caller emits the same wire shape; matches the
    :class:`decoding_sandbox.web.schemas.UsageEvent` model field-for-field
    so a populated sink dumps cleanly onto the SSE wire.
    """
    return {
        "requests": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "notes": [],
    }


def record_request(sink: UsageSink | None, n: int = 1) -> None:
    """Add ``n`` to ``sink["requests"]`` (no-op when ``sink is None``).

    Counts each HTTP attempt -- including retried-then-succeeded
    attempts -- because the user-facing metric we want is "how much did
    this run press on the provider's RPS budget", not "how many calls
    eventually returned 2xx".
    """
    if sink is None:
        return
    sink["requests"] = int(sink.get("requests") or 0) + int(n)


def record_tokens(
    sink: UsageSink | None,
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> None:
    """Accumulate token counts into ``sink``.

    Each argument is optional so a backend can report only what it
    knows (e.g. chat-streaming providers sometimes return only
    ``completion_tokens``). ``None`` slots stay ``None`` until a
    backend writes the first number; subsequent writes add to it.
    """
    if sink is None:
        return
    if prompt_tokens is not None:
        cur = sink.get("prompt_tokens")
        sink["prompt_tokens"] = int(cur or 0) + int(prompt_tokens)
    if completion_tokens is not None:
        cur = sink.get("completion_tokens")
        sink["completion_tokens"] = int(cur or 0) + int(completion_tokens)
    if total_tokens is not None:
        cur = sink.get("total_tokens")
        sink["total_tokens"] = int(cur or 0) + int(total_tokens)


def add_note(sink: UsageSink | None, text: str) -> None:
    """Append a short human-readable advisory to ``sink["notes"]``.

    Deduplicates: re-adding the same text is a no-op so a method invoked
    multiple times during one run doesn't pile up identical lines.
    """
    if sink is None or not text:
        return
    notes = sink.setdefault("notes", [])
    if isinstance(notes, list) and text not in notes:
        notes.append(text)


__all__ = [
    "UsageSink",
    "UsageAware",
    "make_sink",
    "record_request",
    "record_tokens",
    "add_note",
]
