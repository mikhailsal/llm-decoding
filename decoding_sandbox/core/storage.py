"""Storage preflight: refuse to start heavy work when a disk is nearly full.

On `dsbx-host` the Linux ext4 disk is a sparse .img living on C: (local SSD, tight
on space). Downloading models into `/` grows that .img and can exhaust C:.
This module checks free space on the relevant paths and aborts early with a
clear message instead of failing mid-download.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3


@dataclass
class DiskStatus:
    path: str
    exists: bool
    total_gb: float
    used_gb: float
    free_gb: float
    ok: bool

    @property
    def line(self) -> str:
        if not self.exists:
            return f"{self.path:<14} (not present, skipped)"
        flag = "OK " if self.ok else "LOW"
        return f"{self.path:<14} {flag}  free {self.free_gb:6.1f} GB / {self.total_gb:6.1f} GB"


def check_paths(paths: list[str], min_free_gb: float) -> list[DiskStatus]:
    """Return free-space status for each path that exists.

    Non-existent paths (e.g. ~/.cache/dsbx when running on the client) are reported
    with ``exists=False`` and treated as OK (skipped), so the same config works
    on both machines.
    """
    results: list[DiskStatus] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            results.append(DiskStatus(raw, False, 0.0, 0.0, 0.0, ok=True))
            continue
        usage = shutil.disk_usage(p)
        free_gb = usage.free / GIB
        results.append(
            DiskStatus(
                path=raw,
                exists=True,
                total_gb=usage.total / GIB,
                used_gb=usage.used / GIB,
                free_gb=free_gb,
                ok=free_gb >= min_free_gb,
            )
        )
    return results


class StoragePreflightError(RuntimeError):
    """Raised when a disk is below the configured free-space floor."""


def preflight_or_raise(paths: list[str], min_free_gb: float) -> list[DiskStatus]:
    """Check paths; raise StoragePreflightError if any existing path is too low."""
    statuses = check_paths(paths, min_free_gb)
    low = [s for s in statuses if s.exists and not s.ok]
    if low:
        details = "; ".join(f"{s.path}: {s.free_gb:.1f} GB free" for s in low)
        raise StoragePreflightError(
            f"Insufficient free space (need >= {min_free_gb} GB): {details}. "
            "See the 'Storage layout' section of the plan; do not use R: or S:."
        )
    return statuses
