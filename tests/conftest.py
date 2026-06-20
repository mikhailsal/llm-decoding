"""Pytest configuration: ensure ``tests/`` is importable for sibling modules.

Lets test files import ``from tests.fakes import FakeBackend`` without an
``__init__.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
