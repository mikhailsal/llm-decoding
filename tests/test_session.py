"""Tests for the long-lived session REPL dispatcher.

We exercise ``dispatch_session_line`` directly so no real prompt_toolkit
loop or terminal is needed. The CLI handlers it calls (cmd_inspect, etc)
are stubbed via ``monkeypatch`` so we test the dispatcher's *routing*
behavior, not the downstream commands (those have their own tests).
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from decoding_sandbox.cli import session as session_mod
from decoding_sandbox.cli.session import (
    DispatchResult,
    SessionState,
    dispatch_session_line,
)
from tests.fakes import FakeBackend


def _state(backend=None, *, timing=True) -> SessionState:
    """Build a SessionState backed by a FakeBackend and a buffered console."""
    backend = backend or FakeBackend(name="fake")
    return SessionState(
        cfg=object(),
        backend=backend,
        backend_name="fake",
        backend_model=None,
        console=Console(file=StringIO(), force_terminal=False, color_system=None),
        timing_enabled=timing,
    )


# --------------------------------------------------------------------------- #
# Meta-commands
# --------------------------------------------------------------------------- #
def test_blank_line_returns_ok_without_recording_history() -> None:
    s = _state()
    r = dispatch_session_line(s, "")
    assert r.tag == "ok"
    assert s.history == []


def test_quit_aliases_set_should_quit_flag() -> None:
    for cmd in (":quit", ":exit", ":q"):
        s = _state()
        r = dispatch_session_line(s, cmd)
        assert r.tag == "quit"
        assert r.should_quit is True


def test_help_meta_does_not_quit_and_writes_to_console() -> None:
    s = _state()
    r = dispatch_session_line(s, ":help")
    assert r.tag == "meta"
    out = s.console.file.getvalue()
    assert "session commands" in out
    assert "meta:" in out


def test_caps_meta_prints_capability_banner() -> None:
    s = _state()
    r = dispatch_session_line(s, ":caps")
    assert r.tag == "meta"
    out = s.console.file.getvalue()
    assert "backend:" in out
    assert "fake" in out


def test_timing_meta_toggles_state() -> None:
    s = _state(timing=True)
    dispatch_session_line(s, ":timing off")
    assert s.timing_enabled is False
    dispatch_session_line(s, ":timing on")
    assert s.timing_enabled is True


def test_timing_meta_rejects_unknown_argument() -> None:
    s = _state(timing=True)
    r = dispatch_session_line(s, ":timing whatever")
    assert r.tag == "parse_error"
    assert "unknown" in r.message
    # state shouldn't flip
    assert s.timing_enabled is True


def test_history_meta_lists_prior_lines() -> None:
    s = _state()
    dispatch_session_line(s, ":caps")
    dispatch_session_line(s, ":timing off")
    dispatch_session_line(s, ":history")
    out = s.console.file.getvalue()
    # Both prior lines appear (the current :history call isn't in the listing).
    assert ":caps" in out
    assert ":timing off" in out


# --------------------------------------------------------------------------- #
# Backend switching
# --------------------------------------------------------------------------- #
def test_switch_backend_closes_old_and_replaces(monkeypatch) -> None:
    old = FakeBackend(name="old")
    new = FakeBackend(name="new")

    closed: list[str] = []
    orig_close = old.close

    def _track_close():
        closed.append("old-closed")
        orig_close()

    old.close = _track_close  # type: ignore[method-assign]

    def _fake_build_backend(name, cfg, model=None):
        assert name == "new"
        return new

    monkeypatch.setattr(
        "decoding_sandbox.core.factory.build_backend", _fake_build_backend
    )

    s = _state(backend=old)
    r = dispatch_session_line(s, ":backend new")
    assert r.tag == "meta"
    assert s.backend is new
    assert s.backend_name == "new"
    assert closed == ["old-closed"]


def test_switch_backend_without_args_reports_current(monkeypatch) -> None:
    s = _state()
    r = dispatch_session_line(s, ":backend")
    assert r.tag == "meta"
    out = s.console.file.getvalue()
    assert "current backend" in out


def test_switch_backend_failure_keeps_old_marker() -> None:
    """If build_backend raises, we report and keep the old name set
    (we already closed the old one; that's the documented behavior)."""
    s = _state()

    def _explodes(name, cfg, model=None):
        raise RuntimeError("nope")

    import decoding_sandbox.core.factory as fac

    orig = fac.build_backend
    fac.build_backend = _explodes
    try:
        r = dispatch_session_line(s, ":backend broken")
    finally:
        fac.build_backend = orig
    assert r.tag == "parse_error"
    assert "failed to build backend" in r.message


# --------------------------------------------------------------------------- #
# Subcommand dispatch
# --------------------------------------------------------------------------- #
def test_unknown_command_returns_parse_error() -> None:
    s = _state()
    r = dispatch_session_line(s, "nonexistent stuff")
    assert r.tag == "parse_error"


def test_shlex_parse_error_returns_parse_error_not_crash() -> None:
    s = _state()
    # An unterminated quote -- shlex.split raises ValueError.
    r = dispatch_session_line(s, 'inspect "missing-end-quote')
    assert r.tag == "parse_error"
    assert "shell-parse" in r.message


def test_inspect_dispatch_calls_cmd_inspect_with_session_backend(monkeypatch) -> None:
    s = _state()
    captured: dict = {}

    def _fake_cmd_inspect(args, cfg, *, backend=None, show_banner=True):
        captured.update(
            prompt=args.prompt, watch=args.watch, top_k=args.top_k,
            backend=backend, show_banner=show_banner, no_timing=args.no_timing,
        )
        return 0

    # ``_build_session_parser`` lazily imports cmd_inspect; patch the source
    # symbol first and then rebuild the parser so the patched callable is
    # bound to the inspect subcommand.
    monkeypatch.setattr("decoding_sandbox.cli.app.cmd_inspect", _fake_cmd_inspect)
    parser = session_mod._build_session_parser()

    r = dispatch_session_line(
        s, "inspect 'The weather' --watch ' dry' --top-k 5", parser=parser
    )
    assert r.tag == "ok"
    assert captured["prompt"] == "The weather"
    assert captured["watch"] == [" dry"]
    assert captured["top_k"] == 5
    assert captured["backend"] is s.backend
    assert captured["show_banner"] is False  # session suppresses repeat banners
    assert captured["no_timing"] is False  # session timing is on


def test_inspect_dispatch_with_timing_off_passes_no_timing_true(monkeypatch) -> None:
    s = _state(timing=False)
    captured: dict = {}

    def _fake_cmd_inspect(args, cfg, *, backend=None, show_banner=True):
        captured["no_timing"] = args.no_timing
        return 0

    monkeypatch.setattr("decoding_sandbox.cli.app.cmd_inspect", _fake_cmd_inspect)
    parser = session_mod._build_session_parser()
    dispatch_session_line(s, "inspect 'hi'", parser=parser)
    assert captured["no_timing"] is True


def test_generate_dispatch_threads_sampler_and_max_tokens(monkeypatch) -> None:
    s = _state()
    captured: dict = {}

    def _fake_cmd_generate(args, cfg, *, backend=None, show_banner=True):
        captured.update(
            prompt=args.prompt, sampler=args.sampler, max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
        return 0

    monkeypatch.setattr("decoding_sandbox.cli.app.cmd_generate", _fake_cmd_generate)
    parser = session_mod._build_session_parser()
    dispatch_session_line(
        s, "generate 'once upon a time' --sampler top_p --top-p 0.9 --max-tokens 30",
        parser=parser,
    )
    assert captured == {
        "prompt": "once upon a time",
        "sampler": "top_p",
        "max_tokens": 30,
        "top_p": 0.9,
    }


def test_history_records_executed_lines() -> None:
    s = _state()
    dispatch_session_line(s, ":caps")
    dispatch_session_line(s, ":timing off")
    assert s.history == [":caps", ":timing off"]


def test_dispatch_result_should_quit_only_for_quit_tag() -> None:
    assert DispatchResult("quit").should_quit is True
    for tag in ("ok", "meta", "parse_error", "unknown"):
        assert DispatchResult(tag).should_quit is False  # type: ignore[arg-type]
