"""Tests for the manual-decoding TUI command dispatcher.

These exercise the pure ``dispatch_command`` function (no prompt_toolkit needed)
so the grammar is covered end-to-end. The TUI's render loop is intentionally
untested -- it only formats already-tested results.
"""

from __future__ import annotations

import json

import pytest

from dsbx.cli.manual_tui import CommandResult, dispatch_command
from dsbx.core.manual import ManualSession
from tests.fakes import FakeBackend, cand


def _session(*, can_force: bool = True) -> ManualSession:
    backend = FakeBackend(
        tokens={"P": [1], " forced": [9, 10]},
        pieces={1: "P", 2: "X", 3: "Y", 9: " forced", 10: "."},
        distributions={
            (1,): [cand(2, "X", 0.7, 0), cand(3, "Y", 0.3, 1)],
            (1, 2): [cand(3, "Y", 0.5, 0)],
        },
        can_force_token=can_force,
    )
    return ManualSession(backend, "P", top_k=4)


@pytest.mark.parametrize("raw", ["q", "quit", "exit", "  q  "])
def test_dispatch_quit_variants(raw: str) -> None:
    res = dispatch_command(_session(), raw)
    assert res == CommandResult("quit")
    assert res.should_quit is True


@pytest.mark.parametrize("raw", ["?", "help", "  ? "])
def test_dispatch_help_returns_help_text(raw: str) -> None:
    res = dispatch_command(_session(), raw)
    assert res.tag == "help"
    assert "commands:" in res.message


def test_dispatch_empty_input_picks_greedy() -> None:
    s = _session()
    res = dispatch_command(s, "")
    assert res.tag == "pick"
    assert res.message == "X"
    assert s.generated_ids == [2]


def test_dispatch_numeric_input_picks_by_rank() -> None:
    s = _session()
    res = dispatch_command(s, "1")
    assert res.tag == "pick"
    assert res.message == "Y"
    assert s.generated_ids == [3]


def test_dispatch_rank_out_of_range_returns_pick_error() -> None:
    s = _session()
    res = dispatch_command(s, "99")
    assert res.tag == "pick_error"
    assert "out of range" in res.message
    assert s.generated_ids == []


def test_dispatch_undo_when_empty_returns_undo_empty() -> None:
    res = dispatch_command(_session(), "u")
    assert res.tag == "undo_empty"


def test_dispatch_undo_after_pick_returns_undo() -> None:
    s = _session()
    dispatch_command(s, "0")
    res = dispatch_command(s, "u")
    assert res.tag == "undo"
    assert s.generated_ids == []


def test_dispatch_force_appends_arbitrary_text() -> None:
    s = _session()
    res = dispatch_command(s, "f  forced")
    assert res.tag == "force"
    assert s.generated_ids == [9, 10]
    assert " forced" in res.message


def test_dispatch_force_blocked_when_capability_disallows() -> None:
    s = _session(can_force=False)
    res = dispatch_command(s, "f  forced")
    assert res.tag == "force_blocked"
    assert "cannot force" in res.message
    assert s.generated_ids == []


def test_dispatch_set_top_k_updates_session() -> None:
    s = _session()
    res = dispatch_command(s, "k 9")
    assert res.tag == "set_top_k"
    assert s.top_k == 9


def test_dispatch_set_top_k_clamps_to_at_least_one() -> None:
    s = _session()
    dispatch_command(s, "k -5")
    assert s.top_k == 1


def test_dispatch_bad_top_k_returns_error() -> None:
    s = _session()
    res = dispatch_command(s, "k abc")
    assert res.tag == "bad_top_k"


def test_dispatch_save_and_load_roundtrip(tmp_path) -> None:
    s1 = _session()
    dispatch_command(s1, "0")  # adds id 2
    target = tmp_path / "t.json"

    saved = dispatch_command(s1, f"s {target}")
    assert saved.tag == "save"
    assert target.exists()
    raw = json.loads(target.read_text())
    assert raw["generated_ids"] == [2]

    s2 = _session()
    loaded = dispatch_command(s2, f"l {target}")
    assert loaded.tag == "load"
    assert s2.generated_ids == [2]


def test_dispatch_unknown_command_returns_unknown_tag() -> None:
    res = dispatch_command(_session(), "xyz")
    assert res.tag == "unknown"


def test_command_result_should_quit_only_true_for_quit() -> None:
    assert CommandResult("quit").should_quit is True
    for tag in ("help", "pick", "undo", "force", "unknown"):
        assert CommandResult(tag).should_quit is False
