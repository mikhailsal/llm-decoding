"""FastAPI app that exposes a single in-process Backend over HTTP + SSE.

Design notes worth keeping in mind:

- One ``Backend`` instance per process. The factory ``make_app(backend)``
  owns it for the app's lifetime; the optional ``backend_kind`` tag is
  surfaced in ``/v1/info`` so the client can show ``"hf"`` /
  ``"llamacpp-py"`` next to the capability name.
- A single ``threading.Lock`` serializes every backend call. Several
  backends (notably ``LlamaCppPyBackend``) keep a KV-cache keyed by the
  longest common prefix of consecutive requests; interleaving requests
  from two clients would corrupt that cache and produce silently wrong
  logits. Per-request locking is enough for single-user research use --
  the request rate is human-scale, not RPS-scale.
- Sync handlers (``def`` not ``async def``). FastAPI runs them in
  Starlette's threadpool, which is the correct shape for blocking GPU
  work: we don't want to occupy the event loop while a forward pass runs.
- The SSE endpoint runs the engine's existing ``generate(...)`` generator
  on the threadpool path too -- ``StreamingResponse`` accepts a sync
  iterator and consumes it in a thread, so each yield turns into a flushed
  SSE chunk on the wire.
"""

from __future__ import annotations

import json
import random
import threading
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from decoding_sandbox import __version__
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.engine import generate
from decoding_sandbox.core.samplers import make_sampler
from decoding_sandbox.server import schemas as S


def make_app(backend: Backend, *, backend_kind: str = "unknown") -> FastAPI:
    """Build a FastAPI app that owns ``backend`` for its lifetime.

    ``backend_kind`` is the short tag the CLI used (``hf`` /
    ``llamacpp-py``) -- echoed in ``/v1/info`` so the client can render
    it without parsing the capability name string.
    """
    app = FastAPI(
        title="dsbx server",
        version=__version__,
        description=(
            "HTTP + SSE wrapper over a single in-process decoding backend. "
            "See decoding_sandbox/server/schemas.py for wire types."
        ),
    )
    lock = threading.Lock()

    # ---------------------------------------------------------------- info
    @app.get("/v1/info", response_model=S.InfoResponse)
    def info() -> S.InfoResponse:
        # No backend call -- capabilities is a property -- but we still take
        # the lock briefly to avoid racing a backend swap (a future feature).
        with lock:
            caps = backend.capabilities
            loaded = _detect_loaded_model(backend)
        return S.InfoResponse(
            capabilities=S.capabilities_to_wire(caps),
            engine_version=__version__,
            backend_kind=backend_kind,
            loaded_model=loaded,
        )

    # ------------------------------------------------------- tokenization
    @app.post("/v1/tokenize", response_model=S.TokenizeResponse)
    def tokenize(req: S.TokenizeRequest) -> S.TokenizeResponse:
        with lock:
            ids = backend.tokenize(req.text)
        return S.TokenizeResponse(ids=[int(i) for i in ids])

    @app.post("/v1/detokenize", response_model=S.DetokenizeResponse)
    def detokenize(req: S.DetokenizeRequest) -> S.DetokenizeResponse:
        with lock:
            text = backend.detokenize(list(req.ids))
        return S.DetokenizeResponse(text=text)

    @app.post("/v1/piece", response_model=S.PieceResponse)
    def piece(req: S.PieceRequest) -> S.PieceResponse:
        with lock:
            text = backend.piece(int(req.id))
        return S.PieceResponse(text=text)

    # ---------------------------------------------------------- inference
    @app.post("/v1/next_distribution", response_model=S.WireStepResult)
    def next_distribution(req: S.NextDistributionRequest) -> S.WireStepResult:
        with lock:
            step = backend.next_distribution(list(req.ids), int(req.top_k))
        return S.step_to_wire(step)

    @app.post("/v1/score_prompt", response_model=S.ScorePromptResponse)
    def score_prompt(req: S.ScorePromptRequest) -> S.ScorePromptResponse:
        with lock:
            try:
                steps = backend.score_prompt(
                    req.prompt,
                    top_k=int(req.top_k),
                    watch_ids=list(req.watch_ids),
                    prepend_token_ids=list(req.prepend_token_ids),
                )
            except NotImplementedError as exc:
                # OpenAICompatBackend raises this for chat-only providers
                # or unsupported features (e.g. prepend_token_ids on
                # cloud backends that tokenize server-side from a plain
                # prompt string). In practice we don't host those on
                # the server, but keep the error path explicit so future
                # backends behave too.
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return S.ScorePromptResponse(steps=[S.step_to_wire(s) for s in steps])

    @app.post("/v1/verify_greedy", response_model=S.VerifyGreedyResponse)
    def verify_greedy(req: S.VerifyGreedyRequest) -> S.VerifyGreedyResponse:
        verify_fn = getattr(backend, "verify_greedy", None)
        if verify_fn is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"backend {backend.capabilities.name!r} does not implement "
                    "verify_greedy; speculative decoding is not supported here."
                ),
            )
        with lock:
            accepted, correction = verify_fn(list(req.context_ids), list(req.draft_ids))
        return S.VerifyGreedyResponse(
            accepted=int(accepted),
            correction=S.candidate_to_wire(correction) if correction is not None else None,
        )

    # ------------------------------------------------ SSE generate stream
    @app.post("/v1/generate/stream")
    def generate_stream(req: S.GenerateRequest) -> StreamingResponse:
        """Run the engine's generate loop and stream each step as SSE.

        We don't decode incrementally on the server -- ``generate`` already
        yields one ``GenStep`` per token, which we re-encode as a JSON SSE
        event. The wire format is plain ``text/event-stream`` (one
        ``data: <json>\\n\\n`` per event); a terminating ``done`` event
        carries the final ``stop_reason`` even on error so clients always
        see a clean end.
        """
        sampler = _build_sampler_or_400(req.sampler)
        # Snapshot the request now so the streaming generator doesn't
        # accidentally close over a pydantic model that FastAPI might
        # invalidate by the time the body actually runs (it won't in
        # practice, but the snapshot also keeps types tight).
        params = dict(
            prompt=req.prompt,
            max_tokens=int(req.max_tokens),
            top_k=int(req.top_k),
            stop_ids=tuple(int(i) for i in req.stop_ids),
            seed=int(req.seed),
            respect_eos=bool(req.respect_eos),
            watch_ids=tuple(int(i) for i in (req.watch_ids or [])),
            prefix_token_ids=tuple(int(i) for i in (req.prefix_token_ids or [])),
        )

        def event_stream() -> Iterator[bytes]:
            yield from _run_generate_stream(backend, lock, sampler, **params)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disables proxy buffering (nginx)
            },
        )

    # Friendly JSON 404 for the (very common) typo of hitting ``/`` from
    # the browser -- FastAPI's default HTML page is a worse signal.
    @app.get("/")
    def root() -> JSONResponse:
        return JSONResponse(
            {
                "name": "dsbx server",
                "version": __version__,
                "backend_kind": backend_kind,
                "endpoints": [
                    "/v1/info",
                    "/v1/tokenize",
                    "/v1/detokenize",
                    "/v1/piece",
                    "/v1/next_distribution",
                    "/v1/score_prompt",
                    "/v1/verify_greedy",
                    "/v1/generate/stream",
                ],
            }
        )

    return app


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _detect_loaded_model(backend: Backend) -> str | None:
    """Best-effort 'what model is loaded?' across backends.

    Each backend records this differently (``HFBackend.loaded_model``,
    ``LlamaCppPyBackend.model_path``, ``OpenAICompatBackend.model``); rather
    than introduce a method on the protocol just for this, we sniff the
    handful of attributes used today. Returns ``None`` when nothing matches
    so the wire field stays optional.
    """
    for attr in ("loaded_model", "model_path", "model"):
        v = getattr(backend, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def _build_sampler_or_400(spec: S.SamplerSpec):
    """Construct a builtin sampler from the wire spec, mapping errors to 400."""
    if spec.name == "custom":
        raise HTTPException(
            status_code=400,
            detail=(
                "custom samplers cannot run server-side (no remote code "
                "execution); the client should use the per-step "
                "/v1/next_distribution loop instead."
            ),
        )
    # Drop None values so make_sampler() sees only explicit overrides; this
    # matches the CLI's behaviour in cmd_generate (it filters None too).
    params: dict[str, Any] = {k: v for k, v in (spec.params or {}).items() if v is not None}
    try:
        return make_sampler(spec.name, **params)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"unknown sampler: {exc}") from exc
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"bad sampler params: {exc}") from exc


def _sse(payload: dict) -> bytes:
    """Encode one SSE ``data:`` frame with a trailing blank line.

    We send the entire payload on a single ``data:`` line (JSON has no
    embedded newlines after json.dumps with default options), which keeps
    parser logic on the client trivial.
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _run_generate_stream(
    backend: Backend,
    lock: threading.Lock,
    sampler,
    *,
    prompt: str,
    max_tokens: int,
    top_k: int,
    stop_ids: tuple[int, ...],
    seed: int,
    respect_eos: bool,
    watch_ids: tuple[int, ...] = (),
    prefix_token_ids: tuple[int, ...] = (),
) -> Iterator[bytes]:
    """Drive ``core.engine.generate`` and yield SSE bytes per step.

    The whole generation runs while holding ``lock`` so a concurrent
    inspect/score_prompt can't corrupt the backend's KV cache mid-decode.
    Any exception is reported as a final ``done`` event with ``error=...``
    rather than a hard HTTP 500 -- by the time we start streaming we've
    already committed headers, so an exception mid-flight has no other way
    to reach the client cleanly.
    """
    rng = random.Random(seed)
    last_reason: str | None = None
    with lock:
        try:
            for gs in generate(
                backend,
                prompt,
                sampler,
                max_tokens=max_tokens,
                top_k=top_k,
                rng=rng,
                stop_ids=stop_ids,
                respect_eos=respect_eos,
                watch_ids=watch_ids,
                prefix_token_ids=prefix_token_ids,
            ):
                event = S.StepEvent(step=S.genstep_to_wire(gs))
                yield _sse(event.model_dump())
                last_reason = gs.stop_reason
        except Exception as exc:  # noqa: BLE001
            yield _sse(S.DoneEvent(stop_reason=last_reason, error=str(exc)).model_dump())
            return
    yield _sse(S.DoneEvent(stop_reason=last_reason).model_dump())


__all__ = ["make_app"]
