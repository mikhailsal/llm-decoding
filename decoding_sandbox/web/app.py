"""FastAPI application factory for the dsbx web middleware.

This is the single entry point a deployment binds to. ``make_web_app`` takes
a :class:`Config`, a bearer ``token``, and an optional ``frontend_dist`` path
pointing at the built SvelteKit bundle to static-serve.

What the app does:

- Mounts a public, unauthenticated ``GET /api/v1/health`` for liveness probes.
- Mounts every other ``/api/v1/*`` route behind :func:`make_require_bearer`.
- Owns one :class:`BackendRegistry` and one :class:`ManualSessionRegistry` for
  the whole app lifetime; both are closed on shutdown.
- Forwards SSE streams from :mod:`decoding_sandbox.web.streaming` unmodified.
- (Optionally) serves a static SvelteKit bundle at ``/`` so the whole
  deployment runs from one process.

What the app does NOT do:

- It does NOT log full request bodies (they may contain prompts). It logs only
  routes + status codes.
- It does NOT echo internal addresses or stack traces in error responses; the
  global exception handler renders friendly ``{detail}`` JSON.

This module is intentionally a bit long but each route handler is tiny, with
the heavy lifting living in :mod:`backends`, :mod:`sessions`, and
:mod:`streaming`.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from decoding_sandbox import __version__
from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.config import Config
from decoding_sandbox.core.samplers import make_sampler
from decoding_sandbox.core.types import StepResult
from decoding_sandbox.server.schemas import step_to_wire
from decoding_sandbox.web import schemas as S
from decoding_sandbox.web.auth import AuthConfig, make_require_bearer
from decoding_sandbox.web.backends import BackendRegistry
from decoding_sandbox.web.sessions import (
    ManualSessionRegistry,
    load_transcript_into_session,
    transcript_to_dict,
)
from decoding_sandbox.web.streaming import stream_generate, stream_spec

log = logging.getLogger("decoding_sandbox.web.app")


def make_web_app(
    cfg: Config,
    *,
    token: str,
    server_label: str = "dsbx-web",
    cors_origins: list[str] | None = None,
    frontend_dist: str | Path | None = None,
    manual_ttl_seconds: float = 3600.0,
) -> FastAPI:
    """Build the FastAPI app. See module docstring for the full contract."""
    auth = AuthConfig(token=token, server_label=server_label)
    require_bearer = make_require_bearer(auth)
    registry = BackendRegistry(cfg)
    sessions = ManualSessionRegistry(ttl_seconds=manual_ttl_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Startup is a no-op (we built ``registry`` already so the app is
        # ready to serve immediately). Shutdown closes every backend the
        # registry ever loaded so e.g. httpx clients to dsbx-host get a clean
        # disconnect and HF models release VRAM.
        yield
        log.info("dsbx-web: shutting down -- closing %d backends", len(registry.names()))
        registry.close_all()

    app = FastAPI(
        title="dsbx web",
        version=__version__,
        description=(
            "Browser-facing middleware over dsbx backends. "
            "All /api/v1/* routes require Authorization: Bearer <token>, "
            "except /api/v1/health."
        ),
        lifespan=lifespan,
    )

    # CORS: same-origin always works; in dev the frontend lives on the Vite
    # server (typically http://localhost:5173). Anything else must be added
    # explicitly via the [web].cors_origins config entry.
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    # Stash registries on app.state so shutdown can find them even if the
    # closures below have been GC'd by FastAPI's internals.
    app.state.registry = registry
    app.state.sessions = sessions
    app.state.cfg = cfg
    app.state.auth = auth

    # --------------------------------------------------------------- health
    @app.get("/api/v1/health", response_model=S.HealthResponse, tags=["meta"])
    def health() -> S.HealthResponse:
        return S.HealthResponse(ok=True, version=__version__, server_label=server_label)

    # ---------------------------------------------------------------- info
    @app.get(
        "/api/v1/info",
        response_model=S.InfoResponse,
        tags=["meta"],
        dependencies=[Depends(require_bearer)],
    )
    def info() -> S.InfoResponse:
        return S.InfoResponse(
            engine_version=__version__,
            server_label=server_label,
            default_backend=cfg.default_backend,
            backends=registry.list_public(),
        )

    # --------------------------------------------------- tokenize / detok
    @app.post(
        "/api/v1/tokenize",
        response_model=S.TokenizeResponse,
        tags=["backend"],
        dependencies=[Depends(require_bearer)],
    )
    def tokenize(req: S.TokenizeRequest) -> S.TokenizeResponse:
        with _use_backend(registry, req.backend, model=req.model) as backend:
            ids = backend.tokenize(req.text)
        return S.TokenizeResponse(ids=[int(i) for i in ids])

    @app.post(
        "/api/v1/detokenize",
        response_model=S.DetokenizeResponse,
        tags=["backend"],
        dependencies=[Depends(require_bearer)],
    )
    def detokenize(req: S.DetokenizeRequest) -> S.DetokenizeResponse:
        with _use_backend(registry, req.backend, model=req.model) as backend:
            text = backend.detokenize(list(req.ids))
        return S.DetokenizeResponse(text=text)

    @app.post(
        "/api/v1/piece",
        response_model=S.PieceResponse,
        tags=["backend"],
        dependencies=[Depends(require_bearer)],
    )
    def piece(req: S.PieceRequest) -> S.PieceResponse:
        with _use_backend(registry, req.backend, model=req.model) as backend:
            text = backend.piece(int(req.id))
        return S.PieceResponse(text=text)

    # ------------------------------------------------------------ inspect
    @app.post(
        "/api/v1/inspect",
        response_model=S.InspectResponse,
        tags=["inspect"],
        dependencies=[Depends(require_bearer)],
    )
    def inspect(req: S.InspectRequest) -> S.InspectResponse:
        with _use_backend(registry, req.backend, model=req.model) as backend:
            watches = _resolve_watches(
                backend,
                texts=list(req.watch_texts or []),
                ids=list(req.watch_ids or []),
                eos=bool(req.watch_eos),
            )
            caps = backend.capabilities
            # Chat-only cloud paths: we fall back to a next-token distribution
            # row exactly like the TUI does (see ``cmd_inspect``).
            generated_only = (
                backend.__class__.__name__ == "OpenAICompatBackend" and not caps.prompt_logprobs
            )
            if generated_only:
                prompt_tokens = backend.tokenize(req.prompt)
                step = backend.next_distribution(prompt_tokens, top_k=int(req.top_k))
                step.context_text = req.prompt
                step.watched = {w.token_id: backend.lookup_watch(step, w.token_id) for w in watches}
                steps = [step]
                note = (
                    "this backend cannot score prompt tokens; showing the "
                    "next-token distribution after the prompt instead"
                )
            else:
                steps = backend.score_prompt(
                    req.prompt,
                    top_k=int(req.top_k),
                    watch_ids=[w.token_id for w in watches],
                )
                note = ""
            return S.InspectResponse(
                steps=[step_to_wire(s) for s in steps],
                watches=watches,
                is_full_vocab=caps.full_vocab,
                prompt_logprobs=caps.prompt_logprobs,
                note=note,
            )

    # ----------------------------------------------------------- generate
    @app.post(
        "/api/v1/generate/stream",
        tags=["generate"],
        dependencies=[Depends(require_bearer)],
    )
    def generate_stream(req: S.GenerateRequest) -> StreamingResponse:
        sampler_spec = req.sampler
        if sampler_spec.name == "custom":
            raise HTTPException(
                status_code=400,
                detail=(
                    "custom samplers cannot run on the middleware (no remote "
                    "code execution); use a builtin sampler instead."
                ),
            )
        params = {k: v for k, v in (sampler_spec.params or {}).items() if v is not None}
        try:
            sampler = make_sampler(sampler_spec.name, **params)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"unknown sampler: {exc}") from exc
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"bad sampler params: {exc}") from exc

        # Resolve stop_texts against the backend's tokenizer right now so we
        # can bail with a 400 if a stop string isn't single-token, mirroring
        # the CLI's behaviour. We need a backend handle but we shouldn't hold
        # the lock for the whole stream when we're going to acquire it again
        # in stream_generate -- so we tokenize under the lock, release, then
        # stream (the registry's per-backend lock is re-entered there).
        backend_holder = _use_backend(registry, req.backend, model=req.model)
        with backend_holder as backend:
            stop_ids = list(req.stop_ids or [])
            for s in req.stop_texts or []:
                ids = backend.tokenize(s)
                if len(ids) == 1:
                    stop_ids.append(ids[0])

        # Now start the stream. The streaming body acquires the lock again
        # for its duration so the stream is internally consistent.
        def _body():
            with _use_backend(registry, req.backend, model=req.model) as backend:
                yield from stream_generate(
                    backend,
                    prompt=req.prompt,
                    sampler=sampler,
                    sampler_name=sampler_spec.name,
                    sampler_params=params,
                    max_tokens=int(req.max_tokens),
                    top_k=int(req.top_k),
                    stop_ids=[int(i) for i in stop_ids],
                    seed=int(req.seed),
                    respect_eos=bool(req.respect_eos),
                    include_prompt=bool(req.include_prompt),
                )

        return StreamingResponse(
            _body(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------ manual sessions
    @app.post(
        "/api/v1/manual/sessions",
        response_model=S.ManualSnapshot,
        tags=["manual"],
        dependencies=[Depends(require_bearer)],
    )
    def manual_create(req: S.ManualCreateRequest) -> S.ManualSnapshot:
        with _use_backend(registry, req.backend, model=req.model) as backend:
            entry = sessions.create(
                req.backend, backend, req.prompt, top_k=int(req.top_k), model=req.model
            )
            return _snapshot(backend, entry)

    @app.get(
        "/api/v1/manual/sessions/{sid}",
        response_model=S.ManualSnapshot,
        tags=["manual"],
        dependencies=[Depends(require_bearer)],
    )
    def manual_get(sid: str) -> S.ManualSnapshot:
        entry = _get_session_or_404(sessions, sid)
        with _use_backend(registry, entry.backend_name, model=entry.model) as backend, entry.lock:
            return _snapshot(backend, entry)

    @app.post(
        "/api/v1/manual/sessions/{sid}/pick",
        response_model=S.ManualSnapshot,
        tags=["manual"],
        dependencies=[Depends(require_bearer)],
    )
    def manual_pick(sid: str, req: S.ManualPickRequest) -> S.ManualSnapshot:
        entry = _get_session_or_404(sessions, sid)
        with _use_backend(registry, entry.backend_name, model=entry.model) as backend, entry.lock:
            try:
                entry.pick(int(req.rank))
            except IndexError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return _snapshot(backend, entry)

    @app.post(
        "/api/v1/manual/sessions/{sid}/force",
        response_model=S.ManualSnapshot,
        tags=["manual"],
        dependencies=[Depends(require_bearer)],
    )
    def manual_force(sid: str, req: S.ManualForceRequest) -> S.ManualSnapshot:
        if (req.text is None) == (req.id is None):
            raise HTTPException(
                status_code=400,
                detail="force requires exactly one of {text, id}",
            )
        entry = _get_session_or_404(sessions, sid)
        with _use_backend(registry, entry.backend_name, model=entry.model) as backend, entry.lock:
            if not backend.capabilities.can_force_token:
                raise HTTPException(
                    status_code=400,
                    detail=(f"backend {backend.capabilities.name!r} cannot force arbitrary tokens"),
                )
            if req.text is not None:
                entry.force_text(req.text)
            else:
                entry.force_id(int(req.id))
            return _snapshot(backend, entry)

    @app.post(
        "/api/v1/manual/sessions/{sid}/undo",
        response_model=S.ManualSnapshot,
        tags=["manual"],
        dependencies=[Depends(require_bearer)],
    )
    def manual_undo(sid: str) -> S.ManualSnapshot:
        entry = _get_session_or_404(sessions, sid)
        with _use_backend(registry, entry.backend_name, model=entry.model) as backend, entry.lock:
            entry.undo()
            return _snapshot(backend, entry)

    @app.post(
        "/api/v1/manual/sessions/{sid}/set_top_k",
        response_model=S.ManualSnapshot,
        tags=["manual"],
        dependencies=[Depends(require_bearer)],
    )
    def manual_set_top_k(sid: str, req: S.ManualSetTopKRequest) -> S.ManualSnapshot:
        if req.top_k < 1:
            raise HTTPException(status_code=400, detail="top_k must be >= 1")
        entry = _get_session_or_404(sessions, sid)
        with _use_backend(registry, entry.backend_name, model=entry.model) as backend, entry.lock:
            entry.session.top_k = int(req.top_k)
            return _snapshot(backend, entry)

    @app.get(
        "/api/v1/manual/sessions/{sid}/transcript",
        response_model=S.ManualTranscript,
        tags=["manual"],
        dependencies=[Depends(require_bearer)],
    )
    def manual_transcript(sid: str) -> S.ManualTranscript:
        entry = _get_session_or_404(sessions, sid)
        with _use_backend(registry, entry.backend_name, model=entry.model) as backend, entry.lock:
            data = transcript_to_dict(entry, backend=backend)
        return S.ManualTranscript(**data)

    @app.post(
        "/api/v1/manual/sessions/{sid}/load",
        response_model=S.ManualSnapshot,
        tags=["manual"],
        dependencies=[Depends(require_bearer)],
    )
    def manual_load(sid: str, payload: S.ManualTranscript) -> S.ManualSnapshot:
        entry = _get_session_or_404(sessions, sid)
        with _use_backend(registry, entry.backend_name, model=entry.model) as backend, entry.lock:
            load_transcript_into_session(entry, payload.model_dump())
            return _snapshot(backend, entry)

    @app.delete(
        "/api/v1/manual/sessions/{sid}",
        tags=["manual"],
        dependencies=[Depends(require_bearer)],
    )
    def manual_delete(sid: str) -> JSONResponse:
        existed = sessions.delete(sid)
        return JSONResponse({"deleted": existed})

    # ------------------------------------------------------- spec stream
    @app.post(
        "/api/v1/spec/stream",
        tags=["spec"],
        dependencies=[Depends(require_bearer)],
    )
    def spec_stream(req: S.SpecRequest) -> StreamingResponse:
        # Speculative needs two HF backends. We build (and close) them per
        # request -- they pin VRAM and the user should pay for that only
        # when they actively use spec.
        target_name = req.target_backend
        draft_name = req.draft_backend
        if target_name == draft_name:
            raise HTTPException(
                status_code=400,
                detail="target_backend and draft_backend must differ",
            )

        # Build now so we can return a 400 (instead of a half-streamed body)
        # if the names are unknown.
        try:
            target_backend = registry.ensure_loaded(target_name, model=req.target_model)
            draft_backend = registry.ensure_loaded(draft_name, model=req.draft_model)
        except LookupError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"unknown backend: {exc}") from exc

        def _body():
            target_entry = registry.get(target_name)
            draft_entry = registry.get(draft_name)
            # Acquire both locks in a deterministic order to avoid deadlock
            # if a future request also wants both backends.
            first, second = sorted([target_entry, draft_entry], key=lambda e: e.name)
            with first.lock, second.lock:
                yield from stream_spec(
                    target_backend,
                    draft_backend,
                    prompt=req.prompt,
                    gamma=int(req.gamma),
                    max_tokens=int(req.max_tokens),
                )

        return StreamingResponse(
            _body(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------- probe
    _probe_cache: dict[str, object] = {"rows": None, "at": None}

    @app.get(
        "/api/v1/probe",
        response_model=S.ProbeResponse,
        tags=["meta"],
        dependencies=[Depends(require_bearer)],
    )
    def probe(refresh: bool = Query(default=False)) -> S.ProbeResponse:
        from decoding_sandbox.core import provider_probe

        if not refresh and _probe_cache["rows"] is not None:
            return S.ProbeResponse(
                rows=list(_probe_cache["rows"]),  # type: ignore[arg-type]
                fresh=False,
                cached_at=float(_probe_cache["at"]),  # type: ignore[arg-type]
            )
        rows: list[S.ProbeRow] = []
        for name in sorted(cfg.providers):
            r = provider_probe.probe_provider(cfg.provider(name), None)
            rows.append(
                S.ProbeRow(
                    provider=r.provider,
                    model=r.model,
                    chat_logprobs=r.chat_logprobs,
                    prompt_logprobs=r.prompt_logprobs,
                )
            )
        _probe_cache["rows"] = rows
        _probe_cache["at"] = time.time()
        return S.ProbeResponse(rows=rows, fresh=True, cached_at=float(_probe_cache["at"]))

    # ----------------------------------------------------- global errors
    @app.exception_handler(LookupError)
    def _lookup_error(_request: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    def _key_error(_request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    # ------------------------------------------- optional static mount
    if frontend_dist is not None:
        dist_path = Path(frontend_dist).expanduser().resolve()
        if dist_path.is_dir():
            _mount_spa_bundle(app, dist_path)
            log.info("dsbx-web: serving frontend bundle from %s", dist_path)
        else:
            log.warning(
                "dsbx-web: --frontend-dist=%s is not a directory; ignoring",
                dist_path,
            )

    return app


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _use_backend(registry: BackendRegistry, name: str, model: str | None = None):
    """``with _use_backend(...) as backend:`` -- lock + load.

    ``model`` is honored only for cloud providers (see
    :meth:`BackendRegistry.use`); other families ignore it but won't error,
    so callers can pass the request's ``model`` field through without
    branching on family.
    """
    try:
        return registry.use(name, model=model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _get_session_or_404(registry: ManualSessionRegistry, sid: str):
    try:
        return registry.get(sid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _snapshot(backend: Backend, entry) -> S.ManualSnapshot:
    """Build a :class:`ManualSnapshot` from the current session state.

    Holding the per-session lock is the caller's responsibility -- the lock
    keeps ``distribution()`` consistent with the ids the session has just
    appended.
    """
    sess = entry.session
    dist: StepResult = sess.distribution()
    text = backend.detokenize(list(sess.generated_ids)) if sess.generated_ids else ""
    # Pad probs to match ids in case a transcript-load left them short.
    probs: list[float | None] = list(entry.generated_probs)
    while len(probs) < len(sess.generated_ids):
        probs.append(None)
    probs = probs[: len(sess.generated_ids)]
    pieces: list[str] = [backend.piece(int(tid)) for tid in sess.generated_ids]
    return S.ManualSnapshot(
        session_id=entry.session_id,
        backend=entry.backend_name,
        prompt=sess.prompt,
        prompt_ids=list(sess.prompt_ids),
        generated_ids=list(sess.generated_ids),
        generated_text=text,
        top_k=int(sess.top_k),
        distribution=step_to_wire(dist),
        can_force_token=bool(backend.capabilities.can_force_token),
        generated_probs=probs,
        generated_pieces=pieces,
        model=getattr(entry, "model", None),
    )


def _resolve_watches(
    backend: Backend, *, texts: list[str], ids: list[int], eos: bool
) -> list[S.ResolvedWatch]:
    """Resolve --watch / --watch-id / --watch-eos to ResolvedWatch entries.

    Mirrors :func:`decoding_sandbox.cli.app._collect_watch_targets` but does
    *not* print warnings; instead, multi-token text watches silently use
    their first token id (so the UI can still render a column rather than
    nothing). The renderer can hint that case via ``piece`` text.
    """
    out: list[S.ResolvedWatch] = []
    seen: set[int] = set()
    for txt in texts:
        toks = backend.tokenize(txt)
        if not toks:
            continue
        tid = int(toks[0])
        if tid in seen:
            continue
        seen.add(tid)
        out.append(
            S.ResolvedWatch(
                label=f"text:{txt!r}",
                token_id=tid,
                source="text",
                piece=backend.piece(tid) if hasattr(backend, "piece") else "",
            )
        )
    for raw in ids:
        try:
            tid = int(raw)
        except (TypeError, ValueError):
            continue
        if tid in seen:
            continue
        seen.add(tid)
        piece = backend.piece(tid) if hasattr(backend, "piece") else ""
        suffix = f" {piece!r}" if piece else ""
        out.append(
            S.ResolvedWatch(label=f"id={tid}{suffix}", token_id=tid, source="id", piece=piece)
        )
    if eos:
        for tid in backend.capabilities.eos_token_ids:
            if int(tid) in seen:
                continue
            seen.add(int(tid))
            out.append(
                S.ResolvedWatch(
                    label=f"EOS:{int(tid)}",
                    token_id=int(tid),
                    source="eos",
                    piece=backend.piece(int(tid)) if hasattr(backend, "piece") else "",
                )
            )
    return out


def _mount_spa_bundle(app: FastAPI, dist_path: Path) -> None:
    """Mount the SvelteKit static bundle with SPA-aware fallback.

    The ``adapter-static`` output is a flat directory:

        index.html
        inspect.html
        generate.html
        _app/...assets...

    FastAPI's ``StaticFiles`` returns 404 for a request like ``GET /inspect``
    because there's no file at that exact path. The bundle DOES contain
    ``inspect.html`` for that route, plus ``index.html`` as a SvelteKit
    fallback for client-side navigation. We add a small catch-all GET
    handler that:

    1. Returns the literal file if it exists (assets, favicon, etc.).
    2. Otherwise tries ``<path>.html`` (the prerendered per-route file).
    3. Otherwise serves ``index.html`` (SPA fallback).

    The ``/_app/*`` assets are served by mounting StaticFiles at that
    exact prefix; everything else hits the catch-all.
    """
    from fastapi.responses import FileResponse

    app.mount(
        "/_app",
        StaticFiles(directory=str(dist_path / "_app")),
        name="frontend_app",
    )

    index_html = dist_path / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str) -> FileResponse:
        # Reject any path that tries to escape the bundle root.
        candidate_root = dist_path
        target = (candidate_root / full_path).resolve()
        try:
            target.relative_to(candidate_root)
        except ValueError:
            raise HTTPException(status_code=404, detail="not found")

        if target.is_file():
            return FileResponse(str(target))

        # SvelteKit's adapter-static writes one HTML file per route
        # (``inspect.html`` -> route ``/inspect``).
        html_candidate = candidate_root / f"{full_path}.html"
        if html_candidate.is_file():
            return FileResponse(str(html_candidate))

        # Anything else falls through to the SPA's client-side router.
        if not index_html.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(str(index_html))


__all__ = ["make_web_app"]
