"""Manual decoding session: choose every token yourself.

This is the backend-agnostic state machine behind the interactive TUI. Keeping
it separate makes it scriptable and testable without a terminal: you can pick by
rank, force an arbitrary token (even a very unlikely one), undo, and save/load a
transcript -- exactly the operations the TUI exposes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.types import StepResult, TokenCandidate


@dataclass
class ManualSession:
    backend: Backend
    prompt: str
    top_k: int = 12
    prompt_ids: list[int] = field(default_factory=list)
    generated_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            self.prompt_ids = self.backend.tokenize(self.prompt)

    # -- state ------------------------------------------------------------- #
    @property
    def tokens(self) -> list[int]:
        return self.prompt_ids + self.generated_ids

    def distribution(self) -> StepResult:
        """Next-token distribution given everything chosen so far."""
        return self.backend.next_distribution(self.tokens, self.top_k)

    def generated_text(self) -> str:
        return self.backend.detokenize(self.generated_ids) if self.generated_ids else ""

    def full_text(self) -> str:
        return self.backend.detokenize(self.tokens)

    # -- actions ----------------------------------------------------------- #
    def pick(self, rank: int) -> TokenCandidate:
        """Append the candidate at the given rank in the current distribution."""
        cands = self.distribution().candidates
        if rank < 0 or rank >= len(cands):
            raise IndexError(f"rank {rank} out of range (0..{len(cands) - 1})")
        c = cands[rank]
        self.generated_ids.append(c.token_id)
        return c

    def force_text(self, text: str) -> list[TokenCandidate]:
        """Force an arbitrary string (may be multiple tokens). Returns them."""
        ids = self.backend.tokenize(text)
        appended: list[TokenCandidate] = []
        for tid in ids:
            self.generated_ids.append(tid)
            appended.append(
                TokenCandidate(tid, self.backend.piece(tid), logprob=float("nan"), rank=-1)
            )
        return appended

    def force_id(self, token_id: int) -> TokenCandidate:
        self.generated_ids.append(token_id)
        return TokenCandidate(token_id, self.backend.piece(token_id), float("nan"), -1)

    def undo(self) -> int | None:
        """Remove the last generated token. Returns it, or None if empty."""
        if not self.generated_ids:
            return None
        return self.generated_ids.pop()

    # -- persistence ------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "backend": self.backend.capabilities.name,
            "prompt_ids": self.prompt_ids,
            "generated_ids": self.generated_ids,
            "generated_text": self.generated_text(),
            "top_k": self.top_k,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self.prompt = data["prompt"]
        self.prompt_ids = data.get("prompt_ids") or self.backend.tokenize(self.prompt)
        self.generated_ids = list(data.get("generated_ids", []))
        self.top_k = int(data.get("top_k", self.top_k))
