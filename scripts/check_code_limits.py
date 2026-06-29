"""Lightweight code-size guardrails for the decoding-sandbox codebase.

Two checks, deliberately phased so the gate is green *today* while still
nudging the codebase toward smaller units over time:

* File size -- a *hard* gate. Every first-party code file must stay under
  ``FILE_LINE_LIMIT`` lines, except the handful of large modules grandfathered
  in ``LEGACY_FILE_BUDGETS``. A legacy file may not grow past its recorded
  ceiling; the standing TODO is to bring each one under the default limit.
* Function size -- *informational*. Over-long functions are reported but do
  not fail the check. The sandbox has a few orchestration functions
  (streaming loops, argparse wiring, ``make_app`` factories) whose split is a
  follow-up, not a release blocker. Surfacing them keeps the signal visible.

This mirrors the spirit of a strict production code-size gate but with a
lighter, phased policy appropriate for a learning sandbox.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILE_LINE_LIMIT = 700
PYTHON_FUNCTION_LINE_LIMIT = 80

CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx"}

# Directories we never scan: VCS/venv, generated frontend output, vendored
# deps, caches, and the test suite (long, data-heavy tests are fine).
FILE_EXCLUDES = {
    ".git",
    ".venv",
    ".svelte-kit",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
    "__pycache__",
    "tests",
}

# Files already above the default ceiling are grandfathered at their current
# size: touching them must not make them larger. TODO: split each below
# FILE_LINE_LIMIT (tracked as a follow-up issue).
LEGACY_FILE_BUDGETS = {
    "decoding_sandbox/web/app.py": 918,
    "decoding_sandbox/web/backends.py": 840,
    "decoding_sandbox/web/streaming.py": 662,
    "decoding_sandbox/web/schemas.py": 629,
    "decoding_sandbox/server/app.py": 612,
}


def iter_code_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in CODE_EXTENSIONS:
            continue
        if any(part in FILE_EXCLUDES for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return sorted(files)


def check_file_sizes(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        rel_path = path.relative_to(ROOT).as_posix()
        budget = LEGACY_FILE_BUDGETS.get(rel_path)
        if budget is not None:
            if line_count > budget:
                errors.append(
                    f"{rel_path}: file has {line_count} lines; grandfathered "
                    f"ceiling is {budget} (must not grow -- split it instead)"
                )
        elif line_count > FILE_LINE_LIMIT:
            errors.append(f"{rel_path}: file has {line_count} lines; limit is {FILE_LINE_LIMIT}")
    return errors


def check_python_function_sizes(files: list[Path]) -> list[str]:
    warnings: list[str] = []
    for path in files:
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            end_lineno = getattr(node, "end_lineno", None)
            if end_lineno is None:
                continue
            line_count = end_lineno - node.lineno + 1
            if line_count > PYTHON_FUNCTION_LINE_LIMIT:
                rel_path = path.relative_to(ROOT).as_posix()
                warnings.append(
                    f"{rel_path}:{node.lineno} {node.name} has {line_count} lines; "
                    f"soft limit is {PYTHON_FUNCTION_LINE_LIMIT}"
                )
    return warnings


def main() -> int:
    files = iter_code_files()
    errors = check_file_sizes(files)
    warnings = check_python_function_sizes(files)

    if warnings:
        print(f"Function-size advisories ({len(warnings)}; informational, non-blocking):")
        for warning in sorted(warnings):
            print(f"- {warning}")
        print()

    if errors:
        print("Code size checks FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Code size checks passed (file-size gate green).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
