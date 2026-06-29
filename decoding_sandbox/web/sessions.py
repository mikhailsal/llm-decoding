"""In-memory registry of manual-decoding sessions.

The CLI's ``dsbx manual`` is naturally stateful: ``ManualSession`` tracks
``prompt_ids`` + ``generated_ids`` across pick/force/undo operations. The
browser, being stateless on the server side, needs server-side state too --
otherwise every keystroke would have to re-send the whole transcript.

Registry semantics:

- Each session is identified by a UUID4 string returned at creation time.
- The session wraps an existing :class:`decoding_sandbox.core.manual.ManualSession`
  unchanged: every public method still does the work, we just store it.
- A ``threading.Lock`` per session serializes concurrent picks/undos from a
  jittery UI (or two browser tabs sharing the same session token).
- TTL eviction is opportunistic (runs on every ``create`` / ``get``); we use
  ``last_used`` and a configurable max-age so abandoned sessions don't pin
  backend state indefinitely.

Threading model: a single registry-wide lock protects the dict itself; each
session has its own lock for the *contents* of the session. This means
creating a new session never blocks behind a long-running pick on another
session.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.manual import ManualSession
from decoding_sandbox.core.types import TokenCandidate

log = logging.getLogger("decoding_sandbox.web.sessions")

# Hard ceiling on simultaneous sessions; we hit it well before any realistic
# user opens this many tabs, but the cap protects us against accidental
# script loops that create-and-forget. Above the limit ``create`` evicts
# the oldest session.
DEFAULT_MAX_SESSIONS = 64


@dataclass
class _Entry:
    """One row in the registry.

    ``generated_probs`` mirrors ``session.generated_ids`` element-wise:
    each slot is the *linear* probability of the token at the moment the
    user picked it (``exp(logprob)`` clamped to ``[0, 1]``), or ``None``
    when the token was forced and therefore has no associated rank in the
    original distribution. The browser uses this list to color the
    running completion text by per-token confidence -- the same buckets
    the inspect/generate tables use.

    ``model`` is purely informational: the cloud-provider model name the
    session was created with (so the UI can round-trip it in transcripts
    and the load-snapshot reflects what's running).
    """

    session_id: str
    backend_name: str
    session: ManualSession
    lock: threading.Lock
    created_at: float
    last_used: float
    generated_probs: list[float | None] = field(default_factory=list)
    model: str | None = None

    # ------------------------------------------------- mutation helpers
    # These wrap the underlying ManualSession actions so the probs list
    # stays in lockstep with generated_ids. The routes call these (not
    # ``session.pick`` directly) so we have a single place that records
    # the bookkeeping.
    def pick(self, rank: int) -> TokenCandidate:
        cand = self.session.pick(int(rank))
        self.generated_probs.append(_prob_from_logprob(cand.logprob))
        return cand

    def force_text(self, text: str) -> list[TokenCandidate]:
        out = self.session.force_text(text)
        for _ in out:
            self.generated_probs.append(None)
        return out

    def force_id(self, token_id: int) -> TokenCandidate:
        cand = self.session.force_id(int(token_id))
        self.generated_probs.append(None)
        return cand

    def undo(self) -> int | None:
        result = self.session.undo()
        if result is not None and self.generated_probs:
            self.generated_probs.pop()
        return result


def _prob_from_logprob(logprob: float) -> float | None:
    """``exp(logprob)`` -> linear prob, with NaN/inf -> ``None``.

    Mirrors the wire convention: a ``None`` slot means "we don't know the
    probability for this token" (forced tokens or backends that don't
    return a finite logprob for a low-ranked candidate).
    """
    if not math.isfinite(logprob):
        return None
    return float(math.exp(logprob))


class ManualSessionRegistry:
    """UUID-keyed, TTL-evicted store of :class:`ManualSession` instances."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 3600.0,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        now: callable | None = None,  # type: ignore[type-arg]
    ) -> None:
        self._ttl = float(ttl_seconds)
        self._max = int(max_sessions)
        self._now = now or time.time
        self._entries: dict[str, _Entry] = {}
        # Registry-wide lock guards _entries dict mutations only.
        self._mu = threading.Lock()

    # ------------------------------------------------------------------ #
    def create(
        self,
        backend_name: str,
        backend: Backend,
        prompt: str,
        *,
        top_k: int = 12,
        model: str | None = None,
    ) -> _Entry:
        """Create a new session and return its entry.

        We don't run the first ``distribution()`` here -- the caller does it
        under the per-session lock so the create-response and the first state
        snapshot share consistent state.
        """
        self._evict_expired_locked_callers()
        sess = ManualSession(backend, prompt, top_k=int(top_k))
        sid = uuid.uuid4().hex
        now = self._now()
        entry = _Entry(
            session_id=sid,
            backend_name=backend_name,
            session=sess,
            lock=threading.Lock(),
            created_at=now,
            last_used=now,
            model=model,
        )
        with self._mu:
            # Evict the oldest entry if we're at the cap. We choose the
            # oldest by ``last_used`` rather than creation so an actively
            # used long-lived session is preserved.
            while len(self._entries) >= self._max:
                oldest = min(self._entries.values(), key=lambda e: e.last_used)
                log.info(
                    "dsbx-web: evicting oldest manual session %s (cap=%d)",
                    oldest.session_id[:8],
                    self._max,
                )
                del self._entries[oldest.session_id]
            self._entries[sid] = entry
        log.info(
            "dsbx-web: created manual session %s on backend %r",
            sid[:8],
            backend_name,
        )
        return entry

    def get(self, session_id: str) -> _Entry:
        """Fetch a session by id; raises ``KeyError`` if absent or expired."""
        self._evict_expired_locked_callers()
        with self._mu:
            if session_id not in self._entries:
                raise KeyError(f"unknown session {session_id!r}")
            entry = self._entries[session_id]
        entry.last_used = self._now()
        return entry

    def delete(self, session_id: str) -> bool:
        """Drop a session by id. Returns True if it existed."""
        with self._mu:
            existed = self._entries.pop(session_id, None) is not None
        if existed:
            log.info("dsbx-web: deleted manual session %s", session_id[:8])
        return existed

    def iter_entries(self) -> Iterator[_Entry]:
        """Snapshot view (the underlying dict may mutate after iteration)."""
        with self._mu:
            return iter(list(self._entries.values()))

    def __len__(self) -> int:
        with self._mu:
            return len(self._entries)

    # ------------------------------------------------- eviction internals
    def _evict_expired_locked_callers(self) -> None:
        """Remove every session older than ``ttl_seconds`` since ``last_used``.

        Cheap to run on every public call: bounded by ``max_sessions`` <<
        anything important. We take ``self._mu`` briefly, gather the dead
        ids, then drop them.
        """
        if self._ttl <= 0:
            return
        cutoff = self._now() - self._ttl
        with self._mu:
            dead = [sid for sid, e in self._entries.items() if e.last_used < cutoff]
            for sid in dead:
                del self._entries[sid]
        for sid in dead:
            log.info("dsbx-web: TTL-evicting manual session %s", sid[:8])


def transcript_to_dict(entry: _Entry, *, backend: Backend) -> dict:
    """Stable JSON-serializable transcript of an ``_Entry``.

    Mirrors :meth:`ManualSession.to_dict` but always includes the
    *resolved* generated text (the JSON-saving path on the browser writes
    this file verbatim, so the text must already be detokenized).
    """
    sess = entry.session
    return {
        "prompt": sess.prompt,
        "backend": entry.backend_name,
        "prompt_ids": list(sess.prompt_ids),
        "generated_ids": list(sess.generated_ids),
        "generated_text": backend.detokenize(list(sess.generated_ids))
        if sess.generated_ids
        else "",
        "top_k": sess.top_k,
        "model": entry.model,
    }


def load_transcript_into_session(entry: _Entry, payload: dict) -> None:
    """Apply a saved transcript onto an existing session.

    We don't replace the session object (the caller wants the same UUID); we
    just overwrite the relevant fields. The ``backend`` field in the payload
    is informational -- we do NOT switch backends here. The UI is expected
    to pre-select a matching backend before invoking load (mirrors the CLI's
    save/load which is tied to the running session's backend).

    The per-token ``generated_probs`` list is *reset* to ``None`` for every
    loaded token: transcripts are token-id-only by design (the saved file
    doesn't include logprobs), so the UI shows neutral-colored tokens
    until the user picks new ones. Picks made after load will be colored
    normally.
    """
    sess = entry.session
    sess.prompt = str(payload.get("prompt", sess.prompt))
    sess.prompt_ids = [int(i) for i in payload.get("prompt_ids", sess.prompt_ids)]
    sess.generated_ids = [int(i) for i in payload.get("generated_ids", [])]
    if "top_k" in payload:
        with contextlib.suppress(TypeError, ValueError):
            sess.top_k = int(payload["top_k"])
    entry.generated_probs = [None for _ in sess.generated_ids]
    if "model" in payload and payload["model"] is not None:
        entry.model = str(payload["model"])


def write_transcript(path: str | Path, data: dict) -> None:
    """Helper for tests / scripted callers; UI writes to disk client-side."""
    import json

    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


__all__ = [
    "DEFAULT_MAX_SESSIONS",
    "ManualSessionRegistry",
    "load_transcript_into_session",
    "transcript_to_dict",
    "write_transcript",
]
