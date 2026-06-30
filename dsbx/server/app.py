"""FastAPI app that exposes a single in-process Backend over HTTP + SSE.

Design notes worth keeping in mind:

- One model slot per process. The factory ``make_app(...)`` owns a
  :class:`BackendSlot` for the app's lifetime. The slot holds the live
  ``Backend`` (or ``None``), a load-state machine
  (``empty`` -> ``loading`` -> ``ready`` / ``error``), and a ``builder``
  callable so the browser can swap the loaded model without restarting
  the process. The optional ``backend_kind`` tag is surfaced in
  ``/v1/info`` / ``/v1/status`` so the client can show ``"hf"`` /
  ``"llamacpp-py"`` next to the capability name.
- A single ``threading.Lock`` serializes every backend call. Several
  backends (notably ``LlamaCppPyBackend``) keep a KV-cache keyed by the
  longest common prefix of consecutive requests; interleaving requests
  from two clients would corrupt that cache and produce silently wrong
  logits. Per-request locking is enough for single-user research use --
  the request rate is human-scale, not RPS-scale. A model reload also
  acquires that lock before swapping the backend, so an in-flight call
  always finishes against a consistent instance.
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
import logging
import random
import threading
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from dsbx import __version__
from dsbx.core.backend import Backend
from dsbx.core.engine import generate
from dsbx.core.samplers import make_sampler
from dsbx.server import schemas as S

log = logging.getLogger("dsbx.server.app")

# Builder signature: takes the model id/path to load (or ``None`` for the
# kind's default) and returns a fresh, ready-to-serve Backend.
BackendBuilder = Callable[[str | None], Backend]
# Model-lister signature: returns the host catalogue of selectable models.
ModelLister = Callable[[], list[S.ServerModelEntry]]


def _no_builder(_model: str | None) -> Backend:
    raise RuntimeError(
        "this dsbx server was started without a rebuildable backend; "
        "restart `dsbx serve` to change the loaded model."
    )


class BackendSlot:
    """A swappable, state-tracked holder for the server's one heavy backend.

    States:

    - ``empty``   : nothing loaded (a ``--no-preload`` start, or after a
      failed reload that left no working instance).
    - ``loading`` : a background thread is building a backend.
    - ``ready``   : ``backend`` is a live instance serving requests.
    - ``error``   : the most recent load failed; ``error`` carries why.

    Concurrency model: ``lock`` serializes inference (and the backend swap
    at the end of a load). ``_state_lock`` guards the small state fields and
    ensures only one load runs at a time. A reload sets ``loading`` first
    (so new inference calls 409 immediately) and runs the actual build on a
    daemon thread; the heavy GGUF/HF load therefore never blocks the event
    loop or a status poll.
    """

    def __init__(
        self,
        *,
        backend_kind: str,
        builder: BackendBuilder,
        model_lister: ModelLister | None = None,
    ) -> None:
        self.backend_kind = backend_kind
        self._builder = builder
        self._model_lister = model_lister
        self.lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.state: str = "empty"
        self.backend: Backend | None = None
        self.loaded_model: str | None = None
        self.error: str | None = None

    # -- lifecycle ------------------------------------------------------- #
    def adopt(self, backend: Backend, model: str | None) -> None:
        """Install an already-built backend (the eager-preload path)."""
        self.backend = backend
        self.loaded_model = model or _detect_loaded_model(backend)
        self.state = "ready"
        self.error = None

    def start_load(self, model: str | None) -> None:
        """Kick a background (re)load of ``model``. Raises if already loading."""
        with self._state_lock:
            if self.state == "loading":
                raise _AlreadyLoadingError()
            self.state = "loading"
            self.error = None
        t = threading.Thread(target=self._load, args=(model,), name="dsbx-model-load", daemon=True)
        t.start()

    def _load(self, model: str | None) -> None:
        # Close the old backend BEFORE building the new one: the small 6 GB GPU
        # can't hold two 9B models at once, so a "build then swap" would
        # OOM. The trade-off is that a failed reload leaves the slot empty
        # (state=error) rather than falling back to the previous model.
        with self.lock:
            old = self.backend
            self.backend = None
        if old is not None:
            try:
                old.close()
            except Exception as exc:
                log.warning("dsbx server: error closing previous backend: %s", exc)
        try:
            new_backend = self._builder(model)
        except Exception as exc:
            log.exception("dsbx server: model load failed")
            with self._state_lock:
                self.state = "error"
                self.error = str(exc)
                self.loaded_model = None
            return
        with self.lock:
            self.backend = new_backend
        with self._state_lock:
            self.state = "ready"
            self.loaded_model = model or _detect_loaded_model(new_backend)
            self.error = None

    def unload(self) -> None:
        """Unload the current model, leaving the slot empty."""
        with self._state_lock:
            if self.state == "loading":
                raise _AlreadyLoadingError()

        with self.lock:
            old = self.backend
            self.backend = None
        if old is not None:
            try:
                old.close()
            except Exception as exc:
                log.warning("dsbx server: error closing backend on unload: %s", exc)

        with self._state_lock:
            self.state = "empty"
            self.loaded_model = None
            self.error = None

    def close(self) -> None:
        with self._state_lock:
            self.state = "empty"
        with self.lock:
            old = self.backend
            self.backend = None
        if old is not None:
            with suppress(Exception):
                old.close()

    # -- introspection --------------------------------------------------- #
    def status(self) -> S.ServerStatus:
        with self._state_lock:
            state = self.state
            loaded = self.loaded_model
            error = self.error
            backend = self.backend
        caps = None
        if state == "ready" and backend is not None:
            # A ready backend whose ``capabilities`` raises is a genuine
            # server error -- let it propagate to a 500 rather than masking
            # a broken backend as "loaded but featureless".
            caps = S.capabilities_to_wire(backend.capabilities)
        return S.ServerStatus(
            backend_kind=self.backend_kind,
            state=state,
            loaded_model=loaded,
            error=error,
            capabilities=caps,
        )

    def models(self) -> S.ServerModelList:
        entries: list[S.ServerModelEntry] = []
        note = ""
        if self._model_lister is not None:
            try:
                entries = list(self._model_lister())
            except Exception as exc:
                note = f"model discovery failed: {exc.__class__.__name__}"
        else:
            note = "this server does not advertise a model catalogue"
        return S.ServerModelList(backend_kind=self.backend_kind, models=entries, note=note)


class _AlreadyLoadingError(RuntimeError):
    """Internal: raised by ``start_load`` when a load is already in progress."""


def make_app(
    backend: Backend | None = None,
    *,
    backend_kind: str = "unknown",
    builder: BackendBuilder | None = None,
    model: str | None = None,
    model_lister: ModelLister | None = None,
    preload: bool = True,
) -> FastAPI:
    """Build a FastAPI app that owns a swappable model slot.

    Two construction modes:

    - ``make_app(backend, backend_kind=...)`` (legacy / eager): adopt an
      already-built backend into a ``ready`` slot. Pass ``builder=`` too if
      you want ``/v1/reload`` to be able to swap the model.
    - ``make_app(builder=..., model=..., preload=True/False)``: start with
      an empty slot and (optionally) kick a background load of ``model``.
      This is the ``--no-preload`` / lazy path.

    ``backend_kind`` is the short tag the CLI used (``hf`` /
    ``llamacpp-py``) -- echoed in ``/v1/info`` / ``/v1/status`` so the
    client can render it without parsing the capability name string.
    ``model_lister`` returns the host's model catalogue for ``/v1/models``.
    """
    slot = BackendSlot(
        backend_kind=backend_kind,
        builder=builder or _no_builder,
        model_lister=model_lister,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            # Release the GPU/VRAM held by the loaded backend on shutdown.
            slot.close()

    app = FastAPI(
        title="dsbx server",
        version=__version__,
        description=(
            "HTTP + SSE wrapper over a single in-process decoding backend. "
            "See dsbx/server/schemas.py for wire types."
        ),
        lifespan=lifespan,
    )
    if backend is not None:
        slot.adopt(backend, model)
    elif builder is not None and preload:
        slot.start_load(model)
    # Back-compat alias: a few call sites / tests reach for ``app`` then
    # the lock; expose the slot's inference lock under the historical name.
    lock = slot.lock

    # ---------------------------------------------------------------- info
    @app.get("/v1/info", response_model=S.InfoResponse)
    def info() -> S.InfoResponse:
        st = slot.status()
        return S.InfoResponse(
            capabilities=st.capabilities,
            engine_version=__version__,
            backend_kind=backend_kind,
            loaded_model=st.loaded_model,
            state=st.state,
        )

    # -------------------------------------------------------- status/models
    @app.get("/v1/status", response_model=S.ServerStatus)
    def status() -> S.ServerStatus:
        return slot.status()

    @app.get("/v1/models", response_model=S.ServerModelList)
    def models() -> S.ServerModelList:
        return slot.models()

    @app.post("/v1/reload", response_model=S.ServerStatus)
    def reload(req: S.ReloadRequest) -> S.ServerStatus:
        try:
            slot.start_load(req.model)
        except _AlreadyLoadingError as exc:
            raise HTTPException(
                status_code=409,
                detail="a model load is already in progress; poll /v1/status",
            ) from exc
        return slot.status()

    @app.post("/v1/unload", response_model=S.ServerStatus)
    def unload() -> S.ServerStatus:
        try:
            slot.unload()
        except _AlreadyLoadingError as exc:
            raise HTTPException(
                status_code=409,
                detail="a model load is already in progress; poll /v1/status",
            ) from exc
        return slot.status()

    # ------------------------------------------------------- tokenization
    @app.post("/v1/tokenize", response_model=S.TokenizeResponse)
    def tokenize(req: S.TokenizeRequest) -> S.TokenizeResponse:
        with lock:
            backend = _require_ready(slot)
            ids = backend.tokenize(req.text)
        return S.TokenizeResponse(ids=[int(i) for i in ids])

    @app.post("/v1/detokenize", response_model=S.DetokenizeResponse)
    def detokenize(req: S.DetokenizeRequest) -> S.DetokenizeResponse:
        with lock:
            backend = _require_ready(slot)
            text = backend.detokenize(list(req.ids))
        return S.DetokenizeResponse(text=text)

    @app.post("/v1/piece", response_model=S.PieceResponse)
    def piece(req: S.PieceRequest) -> S.PieceResponse:
        with lock:
            backend = _require_ready(slot)
            text = backend.piece(int(req.id))
        return S.PieceResponse(text=text)

    @app.post("/v1/special_tokens", response_model=S.SpecialTokensResponse)
    def special_tokens() -> S.SpecialTokensResponse:
        with lock:
            backend = _require_ready(slot)
            pairs = backend.special_tokens()
        return S.SpecialTokensResponse(
            tokens=[S.SpecialToken(id=int(i), text=str(t)) for i, t in pairs]
        )

    # ---------------------------------------------------------- inference
    @app.post("/v1/next_distribution", response_model=S.WireStepResult)
    def next_distribution(req: S.NextDistributionRequest) -> S.WireStepResult:
        with lock:
            backend = _require_ready(slot)
            step = backend.next_distribution(list(req.ids), int(req.top_k))
        return S.step_to_wire(step)

    @app.post("/v1/score_prompt", response_model=S.ScorePromptResponse)
    def score_prompt(req: S.ScorePromptRequest) -> S.ScorePromptResponse:
        with lock:
            backend = _require_ready(slot)
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
        with lock:
            backend = _require_ready(slot)
            verify_fn = getattr(backend, "verify_greedy", None)
            if verify_fn is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"backend {backend.capabilities.name!r} does not implement "
                        "verify_greedy; speculative decoding is not supported here."
                    ),
                )
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
        # Fail fast with a clean 409 if no model is loaded -- once the
        # StreamingResponse body starts, headers are committed and the
        # only way to report an error is the in-band ``done`` event.
        with lock:
            _require_ready(slot)
        # Snapshot the request now so the streaming generator doesn't
        # accidentally close over a pydantic model that FastAPI might
        # invalidate by the time the body actually runs (it won't in
        # practice, but the snapshot also keeps types tight). ``Any``-
        # typed so the ``**params`` splat below satisfies the per-
        # parameter type signature of ``_run_generate_stream``.
        params: dict[str, Any] = {
            "prompt": req.prompt,
            "max_tokens": int(req.max_tokens),
            "top_k": int(req.top_k),
            "stop_ids": tuple(int(i) for i in req.stop_ids),
            "seed": int(req.seed),
            "respect_eos": bool(req.respect_eos),
            "watch_ids": tuple(int(i) for i in (req.watch_ids or [])),
            "prefix_token_ids": tuple(int(i) for i in (req.prefix_token_ids or [])),
        }

        def event_stream() -> Iterator[bytes]:
            yield from _run_generate_stream(slot, sampler, **params)

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
                    "/v1/status",
                    "/v1/models",
                    "/v1/reload",
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

    app.state.slot = slot
    return app


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_ready(slot: BackendSlot) -> Backend:
    """Return the live backend or raise a clean 409 describing the slot state.

    Callers hold ``slot.lock`` so the ``backend`` they get back can't be
    swapped out from under them mid-call.
    """
    state = slot.state
    backend = slot.backend
    if state == "ready" and backend is not None:
        return backend
    if state == "loading":
        raise HTTPException(status_code=409, detail="model is loading; retry shortly")
    if state == "error":
        raise HTTPException(
            status_code=409,
            detail=f"model failed to load: {slot.error or 'unknown error'}",
        )
    raise HTTPException(
        status_code=409,
        detail="no model loaded; POST /v1/reload to load one",
    )


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
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _run_generate_stream(
    slot: BackendSlot,
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

    The whole generation runs while holding ``slot.lock`` so a concurrent
    inspect/score_prompt -- or a model reload -- can't corrupt the
    backend's KV cache mid-decode. Any exception is reported as a final
    ``done`` event with ``error=...`` rather than a hard HTTP 500 -- by the
    time we start streaming we've already committed headers, so an
    exception mid-flight has no other way to reach the client cleanly.
    """
    rng = random.Random(seed)
    last_reason: str | None = None
    with slot.lock:
        try:
            backend = _require_ready_inline(slot)
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
        except Exception as exc:
            yield _sse(S.DoneEvent(stop_reason=last_reason, error=str(exc)).model_dump())
            return
    yield _sse(S.DoneEvent(stop_reason=last_reason).model_dump())


def _require_ready_inline(slot: BackendSlot) -> Backend:
    """Like :func:`_require_ready` but raises a plain error for the stream path.

    The streaming body has already committed headers, so it converts any
    error into a ``done`` event itself -- here we just need a backend or a
    descriptive exception (never an ``HTTPException``, which Starlette would
    not know how to render mid-stream).
    """
    if slot.state == "ready" and slot.backend is not None:
        return slot.backend
    raise RuntimeError(
        f"no model ready to serve (state={slot.state}"
        + (f": {slot.error}" if slot.error else "")
        + ")"
    )


__all__ = ["BackendSlot", "make_app"]
