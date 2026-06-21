"""Tests for the storage preflight."""

from __future__ import annotations

import pytest

from decoding_sandbox.core import storage


class _FakeUsage:
    def __init__(self, total: float, used: float, free: float) -> None:
        self.total = total
        self.used = used
        self.free = free


def test_check_paths_skips_missing_paths_and_marks_ok(monkeypatch, tmp_path) -> None:
    fake_root = tmp_path
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda p: _FakeUsage(100 * storage.GIB, 50 * storage.GIB, 50 * storage.GIB),
    )

    statuses = storage.check_paths([str(fake_root), "/does/not/exist"], min_free_gb=10.0)

    assert len(statuses) == 2
    assert statuses[0].exists is True
    assert statuses[0].ok is True
    assert statuses[0].free_gb == pytest.approx(50.0)
    assert statuses[1].exists is False
    assert statuses[1].ok is True  # missing -> treated as OK so configs port across machines


def test_check_paths_marks_low_when_below_floor(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda p: _FakeUsage(100 * storage.GIB, 99 * storage.GIB, 1 * storage.GIB),
    )

    [s] = storage.check_paths([str(tmp_path)], min_free_gb=5.0)

    assert s.exists is True
    assert s.ok is False
    assert s.free_gb == pytest.approx(1.0)


def test_preflight_or_raise_passes_when_all_paths_ok(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda p: _FakeUsage(100 * storage.GIB, 50 * storage.GIB, 50 * storage.GIB),
    )

    statuses = storage.preflight_or_raise([str(tmp_path)], min_free_gb=10.0)
    assert statuses[0].ok is True


def test_preflight_or_raise_raises_with_details_when_too_low(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda p: _FakeUsage(10 * storage.GIB, 9 * storage.GIB, 1 * storage.GIB),
    )

    with pytest.raises(storage.StoragePreflightError) as exc_info:
        storage.preflight_or_raise([str(tmp_path)], min_free_gb=5.0)
    assert str(tmp_path) in str(exc_info.value)
    assert "1.0 GB free" in str(exc_info.value)


def test_preflight_or_raise_ignores_missing_paths(monkeypatch) -> None:
    # No real disk calls because nothing exists.
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda p: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    statuses = storage.preflight_or_raise(["/no/such/path"], min_free_gb=5.0)
    assert statuses[0].exists is False
    assert statuses[0].ok is True


def test_disk_status_line_includes_ok_or_low(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda p: _FakeUsage(100 * storage.GIB, 50 * storage.GIB, 50 * storage.GIB),
    )

    [s] = storage.check_paths([str(tmp_path)], min_free_gb=10.0)
    assert "OK" in s.line


def test_disk_status_line_for_missing_path_says_skipped() -> None:
    [s] = storage.check_paths(["/no/such/path/at/all"], min_free_gb=1.0)
    assert "skipped" in s.line.lower() or "not present" in s.line.lower()
