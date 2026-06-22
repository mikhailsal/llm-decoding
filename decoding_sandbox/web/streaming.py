"""SSE-stream helpers for the dsbx web middleware.

Both ``/api/v1/generate/stream`` and ``/api/v1/spec/stream`` produce
``text/event-stream`` bodies. The shape is intentionally identical to what
``decoding_sandbox/server/app.py`` already emits, so the browser's SSE parser
is the same as the one inside ``decoding_sandbox/backends/remote.py``.

Two streaming sources collide here:

1. The middleware is talking to a ``RemoteBackend`` (the common client
   -> dsbx-host case). The remote already runs the engine; we forward each
   ``GenStep`` it yields onto the browser unchanged.
2. The middleware is wrapping an in-process backend (``hf`` / ``llamacpp-py``
   on the same host). We run ``core.engine.generate`` here and emit the
   same wire shape.

We unify both with the helper ``stream_generate(backend, ...)`` which uses
``backend.stream_generate`` when available (the ``supports_remote_stream``
marker on ``RemoteBackend``) and falls back to the in-process loop otherwise.

Spec is server-only for now (HF target/draft), so its helper is simpler.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator
from typing import Any

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.engine import generate as core_generate
from decoding_sandbox.core.samplers import Sampler
from decoding_sandbox.core.speculative import speculative_generate
from decoding_sandbox.server.schemas import genstep_to_wire, step_to_wire

log = logging.getLogger("decoding_sandbox.web.streaming")


def sse_frame(payload: dict) -> bytes:
    """Encode one SSE frame -- ``data: <json>\\n\\n``.

    Identical to the encoder in :mod:`decoding_sandbox.server.app`. Kept
    here as a free function so the streaming routes don't have to import
    a private symbol from the other server.
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def stream_generate(
    backend: Backend,
    *,
    prompt: str,
    sampler: Sampler,
    sampler_name: str,
    sampler_params: dict[str, Any],
    max_tokens: int,
    top_k: int,
    stop_ids: list[int],
    seed: int,
    respect_eos: bool,
    include_prompt: bool = False,
) -> Iterator[bytes]:
    """Yield SSE frames for a generate call against ``backend``.

    Internally we *always* run a per-step loop -- even when the backend is a
    ``RemoteBackend`` that has ``stream_generate``. That keeps the wire
    format produced here byte-identical regardless of whether the underlying
    engine lives on this host or on ``dsbx-host``.

    Why not just forward ``backend.stream_generate`` events directly? Because
    we'd lose the local ability to mutate the wire payload (e.g. scrub a
    field, normalize an error). Using one code path here means the test
    suite can exercise the streaming output deterministically against a
    ``FakeBackend`` and the production path will produce the same shape.

    When ``include_prompt`` is true we emit a single ``prompt_score`` frame
    BEFORE the step events: it carries the per-prompt-token distributions
    (when the backend supports prompt logprobs) or a one-row next-token
    distribution fallback (chat-only cloud paths). The frame's body is
    shape-compatible with :class:`decoding_sandbox.web.schemas.InspectResponse`.
    """
    use_remote_stream = hasattr(backend, "stream_generate")

    last_reason: str | None = None
    try:
        if include_prompt:
            yield from _emit_prompt_score(backend, prompt=prompt, top_k=top_k)
        if use_remote_stream:
            yield from _forward_remote_stream(
                backend,
                prompt=prompt,
                sampler_name=sampler_name,
                sampler_params=sampler_params,
                max_tokens=max_tokens,
                top_k=top_k,
                stop_ids=stop_ids,
                seed=seed,
                respect_eos=respect_eos,
            )
            # _forward_remote_stream owns the terminating ``done`` frame.
            return
        rng = random.Random(seed)
        for gs in core_generate(
            backend,
            prompt,
            sampler,
            max_tokens=max_tokens,
            top_k=top_k,
            rng=rng,
            stop_ids=stop_ids,
            respect_eos=respect_eos,
        ):
            yield sse_frame({"event": "step", "step": genstep_to_wire(gs).model_dump()})
            last_reason = gs.stop_reason
    except Exception as exc:  # noqa: BLE001
        log.exception("dsbx-web: generate stream errored")
        yield sse_frame({"event": "done", "stop_reason": last_reason, "error": str(exc)})
        return
    yield sse_frame({"event": "done", "stop_reason": last_reason})


def _emit_prompt_score(backend: Backend, *, prompt: str, top_k: int) -> Iterator[bytes]:
    """Emit a single ``prompt_score`` SSE frame for the given prompt.

    Mirrors the logic in ``/api/v1/inspect``: backends that can score the
    prompt (HF, llamacpp-py, Fireworks-with-echo, RemoteBackend backed by
    any of those) yield one row per prompt token; chat-only backends fall
    back to a single next-token distribution row so the UI still has
    something useful to render.

    Errors inside this function become a ``prompt_score`` frame with an
    empty steps list and a ``note`` -- the regular generation loop then
    runs to completion, so the user sees "couldn't score prompt" rather
    than a hard 500.
    """
    caps = backend.capabilities
    try:
        chat_only = backend.__class__.__name__ == "OpenAICompatBackend" and not caps.prompt_logprobs
        if chat_only:
            tokens = backend.tokenize(prompt)
            step = backend.next_distribution(tokens, top_k=int(top_k))
            step.context_text = prompt
            steps_wire = [step_to_wire(step).model_dump()]
            note = (
                "this backend cannot score prompt tokens; showing the "
                "next-token distribution after the prompt instead"
            )
        else:
            steps = backend.score_prompt(prompt, top_k=int(top_k), watch_ids=[])
            steps_wire = [step_to_wire(s).model_dump() for s in steps]
            note = ""
    except Exception as exc:  # noqa: BLE001
        log.warning("dsbx-web: prompt scoring failed: %s", exc)
        yield sse_frame(
            {
                "event": "prompt_score",
                "steps": [],
                "is_full_vocab": bool(caps.full_vocab),
                "prompt_logprobs": bool(caps.prompt_logprobs),
                "note": f"prompt scoring failed: {exc}",
            }
        )
        return
    yield sse_frame(
        {
            "event": "prompt_score",
            "steps": steps_wire,
            "is_full_vocab": bool(caps.full_vocab),
            "prompt_logprobs": bool(caps.prompt_logprobs),
            "note": note,
        }
    )


def _forward_remote_stream(
    backend: Backend,
    *,
    prompt: str,
    sampler_name: str,
    sampler_params: dict[str, Any],
    max_tokens: int,
    top_k: int,
    stop_ids: list[int],
    seed: int,
    respect_eos: bool,
) -> Iterator[bytes]:
    """Pull ``GenStep`` objects from a remote backend, re-encode for the wire.

    The remote already framed each step as SSE on the wire; ``RemoteBackend``
    parses those into ``GenStep`` instances for us. We just need to
    re-serialize each one into the wire shape *this* server promises.
    """
    from decoding_sandbox.backends.remote import RemoteBackendError

    last_reason: str | None = None
    try:
        iterator = backend.stream_generate(  # type: ignore[attr-defined]
            prompt,
            sampler_name=sampler_name,
            sampler_params=sampler_params,
            max_tokens=max_tokens,
            top_k=top_k,
            stop_ids=stop_ids,
            seed=seed,
            respect_eos=respect_eos,
        )
        for gs in iterator:
            yield sse_frame({"event": "step", "step": genstep_to_wire(gs).model_dump()})
            last_reason = gs.stop_reason
    except RemoteBackendError as exc:
        log.warning("dsbx-web: remote generate stream errored: %s", exc)
        yield sse_frame({"event": "done", "stop_reason": last_reason, "error": str(exc)})
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("dsbx-web: remote generate stream errored unexpectedly")
        yield sse_frame({"event": "done", "stop_reason": last_reason, "error": str(exc)})
        return
    yield sse_frame({"event": "done", "stop_reason": last_reason})


def stream_spec(
    target: Backend,
    draft: Backend,
    *,
    prompt: str,
    gamma: int,
    max_tokens: int,
) -> Iterator[bytes]:
    """Yield SSE frames for one speculative-decoding pass.

    Emits ``{"event":"round", "round": {...}}`` per round, then a final
    ``{"event":"done", "total_proposed": ..., "completion": "..."}`` summary.
    Errors land in the terminating ``done`` event with an ``error`` field,
    matching ``stream_generate``.
    """
    from decoding_sandbox.server.schemas import candidate_to_wire

    total_proposed = total_accepted = total_emitted = rounds = 0
    all_ids: list[int] = []
    try:
        for rnd in speculative_generate(
            target,
            draft,
            prompt,
            gamma=gamma,
            max_tokens=max_tokens,
        ):
            payload = {
                "event": "round",
                "round": {
                    "step": int(rnd.step),
                    "proposed": [candidate_to_wire(c).model_dump() for c in rnd.proposed],
                    "accepted": int(rnd.accepted),
                    "correction": candidate_to_wire(rnd.correction).model_dump()
                    if rnd.correction is not None
                    else None,
                    "emitted_ids": [int(i) for i in rnd.emitted_ids],
                },
            }
            yield sse_frame(payload)
            total_proposed += len(rnd.proposed)
            total_accepted += rnd.accepted
            total_emitted += len(rnd.emitted_ids)
            rounds += 1
            all_ids.extend(rnd.emitted_ids)
    except Exception as exc:  # noqa: BLE001
        log.exception("dsbx-web: spec stream errored")
        yield sse_frame(
            {
                "event": "done",
                "total_proposed": total_proposed,
                "total_accepted": total_accepted,
                "total_emitted": total_emitted,
                "rounds": rounds,
                "completion": "",
                "error": str(exc),
            }
        )
        return
    completion = target.detokenize(all_ids) if all_ids else ""
    yield sse_frame(
        {
            "event": "done",
            "total_proposed": total_proposed,
            "total_accepted": total_accepted,
            "total_emitted": total_emitted,
            "rounds": rounds,
            "completion": completion,
        }
    )


__all__ = ["sse_frame", "stream_generate", "stream_spec"]
