"""Non-interactive exercise of ManualSession (Wave 3 test).

Drives a short manual decode: greedy pick, a FORCED non-top token, an undo, and
save/load round-trip. Works on any backend.

Run on dsbx-host:
  python scripts/test_manual.py --backend llamacpp
  python scripts/test_manual.py --backend hf
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from decoding_sandbox.core.config import load_config
from decoding_sandbox.core.factory import build_backend
from decoding_sandbox.core.manual import ManualSession


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="llamacpp")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--force", default=" Berlin")
    args = ap.parse_args()

    cfg = load_config()
    backend = build_backend(args.backend, cfg)
    session = ManualSession(backend, args.prompt, top_k=8)

    print(f"backend: {backend.capabilities.name}")
    print(f"prompt : {args.prompt!r}")

    top = session.distribution().candidates[0]
    print(f"\ntop-1 candidate: {top.text!r} p={top.prob:.2%}")

    picked = session.pick(0)
    print(f"picked rank 0 -> {picked.text!r}; text now: {session.generated_text()!r}")

    forced = session.force_text(args.force)
    print(f"forced {args.force!r} -> {[f.text for f in forced]}; text now: {session.generated_text()!r}")

    undone = session.undo()
    print(f"undo -> removed token id {undone}; text now: {session.generated_text()!r}")

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "transcript.json"
        session.save(path)
        ids_before = list(session.generated_ids)
        session2 = ManualSession(backend, "placeholder")
        session2.load(path)
        ok = session2.generated_ids == ids_before
        print(f"\nsave/load round-trip: {'OK' if ok else 'MISMATCH'}")
        print(f"loaded text: {session2.generated_text()!r}")

    backend.close()
    print("\nMANUAL TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
