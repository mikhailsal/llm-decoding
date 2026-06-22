"""In-process llama.cpp backend via the ``llama-cpp-python`` binding.

This is the white-box engine for GGUF models on hardware where HF transformers
won't load them. Specifically: the 9B Qwen3.5 base hybrid arch fails on the 6
GB Pascal P40 under bitsandbytes 4-bit + CPU offload (verified meta-tensor
bug), but its Q4 GGUF runs fine on the same hardware via llama.cpp with
``-ngl 20``. ``llama-cpp-python`` exposes the same engine in-process, and with
``logits_all=True`` we can grab the full ``[seq, vocab]`` logits tensor -- the
exact full-vocab distribution at every position, in a single forward pass.

Capabilities advertised: ``full_vocab=True``, ``prompt_logprobs=True`` --
identical to ``HFBackend``. The implementations of ``next_distribution`` and
``score_prompt`` produce ``StepResult`` values indistinguishable from HF's, so
``inspect``/``generate``/``manual``/``spec`` (via ``verify_greedy``) all work
without any UI changes.

Why a separate backend from ``llamacpp`` (HTTP): the existing HTTP backend is
client-side and convenient when a llama-server is already running for other
tools; this one is an embedded process that controls eval/cache/scores
directly. They share the GGUF on disk -- no extra download.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate


def _discover_model_path(
    explicit: str | None,
    search_dirs: list[str],
    glob: str,
) -> str:
    """Locate the GGUF: explicit path wins, else first match under search_dirs."""
    if explicit:
        p = Path(os.path.expanduser(os.path.expandvars(explicit)))
        if not p.is_file():
            raise FileNotFoundError(f"GGUF not found at {p}")
        return str(p)
    for raw in search_dirs:
        d = Path(os.path.expanduser(os.path.expandvars(raw)))
        if not d.is_dir():
            continue
        for match in d.glob(glob):
            if match.is_file():
                return str(match)
    raise FileNotFoundError(
        f"No GGUF matching {glob!r} under {search_dirs}. "
        "Set [local.llamacpp_py].model_path in config.toml, or run "
        "scripts/setup_wind.sh to download it."
    )


class LlamaCppPyBackend(Backend):
    """Embedded llama.cpp with full-vocab logit access."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        model_glob: str = "**/Qwen3.5-9B-Base-Q4_K_M.gguf",
        model_search_dirs: list[str] | None = None,
        n_gpu_layers: int = 20,
        n_ctx: int = 4096,
        logits_all: bool = True,
        verbose: bool = False,
        **extra_llama_kwargs: Any,
    ) -> None:
        from llama_cpp import Llama  # type: ignore

        import numpy as np  # noqa: F401  (validate availability early)

        self._numpy = __import__("numpy")
        self.model_path = _discover_model_path(model_path, model_search_dirs or [], model_glob)
        self._llama = Llama(
            model_path=self.model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=logits_all,
            verbose=verbose,
            **extra_llama_kwargs,
        )
        self._logits_all = bool(logits_all)
        self._n_vocab = int(self._llama.n_vocab())
        self._piece_cache: dict[int, str] = {}
        # Tracks how many tokens are in the model's KV cache. We can skip
        # re-eval'ing a prefix when the new context is a strict extension.
        self._cached_ids: list[int] = []
        # EOS ids: llama-cpp-python exposes the model's EOS via Llama.token_eos().
        # Some GGUFs (Qwen-style chat models) also tag <|im_end|> as an EOG token
        # (end-of-generation) -- if the binding exposes a list, take all of them;
        # otherwise fall back to the single token_eos().
        self._eos_ids: tuple[int, ...] = self._discover_eos_ids()

    # -- introspection ----------------------------------------------------- #
    @property
    def capabilities(self) -> Capabilities:
        name = Path(self.model_path).stem
        return Capabilities(
            name=f"llamacpp-py:{name}",
            full_vocab=self._logits_all,
            prompt_logprobs=self._logits_all,
            max_top_logprobs=self._n_vocab,
            can_force_token=True,
            notes=(
                "in-process llama.cpp; full-vocab logits via Llama.scores "
                "(logits_all=True), whole-context in one forward pass."
                if self._logits_all
                else "in-process llama.cpp; logits_all=False -> last-position only."
            ),
            eos_token_ids=self._eos_ids,
        )

    def _discover_eos_ids(self) -> tuple[int, ...]:
        """Best-effort EOS extraction from the llama.cpp binding.

        ``Llama.token_eos()`` is the canonical answer; some newer bindings
        also expose ``Llama.token_eot()`` (end-of-turn) for chat templates.
        Anything that raises or returns a negative id is dropped -- a
        negative id from llama.cpp means "no EOS configured".
        """
        out: list[int] = []
        for attr in ("token_eos", "token_eot"):
            fn = getattr(self._llama, attr, None)
            if fn is None:
                continue
            try:
                tid = int(fn())
            except Exception:  # noqa: BLE001
                continue
            if tid >= 0 and tid not in out:
                out.append(tid)
        return tuple(out)

    def _is_special(self, token_id: int) -> bool:
        if token_id in self._eos_ids:
            return True
        # Fallback: many GGUF tokenizers print specials as ``<|...|>``. We
        # use the renderer's heuristic so detection lines up everywhere.
        from decoding_sandbox.cli.render import is_special_text

        return is_special_text(self.piece(token_id))

    # -- tokenization ------------------------------------------------------ #
    def tokenize(self, text: str) -> list[int]:
        # ``add_bos=False`` so token ids align with what the user wrote -- the
        # GGUF's prefilled chat templates aren't relevant for base-model
        # inspection. Result is a list of ints.
        return list(self._llama.tokenize(text.encode("utf-8"), add_bos=False))

    def detokenize(self, token_ids: list[int]) -> str:
        return self._llama.detokenize(list(token_ids)).decode("utf-8", errors="replace")

    def piece(self, token_id: int) -> str:
        if token_id not in self._piece_cache:
            self._piece_cache[token_id] = self._llama.detokenize([token_id]).decode(
                "utf-8", errors="replace"
            )
        return self._piece_cache[token_id]

    # -- core: full-vocab logits over a prompt ----------------------------- #
    def _logsoftmax_all(self, token_ids: list[int]):
        """Run a forward pass and return a [len(ids), vocab] log-softmax matrix.

        Reuses the KV cache when the new context is a strict extension of the
        previously evaluated one, which is the common case for inspect (called
        once on a fresh prompt) and manual decoding (each step appends one or
        two tokens).
        """
        np = self._numpy
        if not self._logits_all:
            raise RuntimeError(
                "LlamaCppPyBackend was constructed with logits_all=False; "
                "full-vocab whole-context inspection is unavailable."
            )

        common = self._common_prefix_len(self._cached_ids, token_ids)
        if common < len(self._cached_ids):
            self._llama.reset()
            common = 0
        new_tokens = token_ids[common:]
        if new_tokens:
            self._llama.eval(new_tokens)
        self._cached_ids = list(token_ids)

        # Llama.scores is a (n_ctx, n_vocab) float32 buffer; only the first
        # `len(token_ids)` rows are populated by our eval.
        scores = np.asarray(self._llama.scores[: len(token_ids)], dtype=np.float32)
        overhead_mb = scores.nbytes / (1024 * 1024)
        if hasattr(self, "_verbose") and self._verbose or True:  # print unconditionally for now, or use rich.print
            import rich
            rich.print(f"[dim]\\[llamacpp-py] logits matrix shape {scores.shape} allocated {overhead_mb:.2f} MB[/dim]")
            
        if scores.shape[0] != len(token_ids):
            raise RuntimeError(
                f"unexpected scores shape {scores.shape} for {len(token_ids)} tokens"
            )
        return _log_softmax(scores, np)

    @staticmethod
    def _common_prefix_len(a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    # -- Backend protocol -------------------------------------------------- #
    def next_distribution(
        self,
        token_ids: list[int],
        top_k: int,
        *,
        watch_ids: Sequence[int] = (),
    ) -> StepResult:
        if not token_ids:
            return StepResult(position=0, candidates=[], is_full_vocab=True)
        np = self._numpy
        logp = self._logsoftmax_all(list(token_ids))[-1]
        k = max(1, min(top_k, logp.shape[-1]))
        idx_part = np.argpartition(-logp, k - 1)[:k]
        order = np.argsort(-logp[idx_part])
        idx = idx_part[order]
        vals = logp[idx]
        cands = [
            TokenCandidate(
                int(j),
                self.piece(int(j)),
                float(v),
                rank,
                is_special=self._is_special(int(j)),
            )
            for rank, (j, v) in enumerate(zip(idx.tolist(), vals.tolist()))
        ]
        step = StepResult(position=len(token_ids), candidates=cands, is_full_vocab=True)
        # Full-vocab backend: read EXACT logprobs for each watched id
        # from the same forward-pass tensor (no separate eval needed),
        # including ids that fell outside the requested top_k.
        for wid in watch_ids:
            wid_i = int(wid)
            if 0 <= wid_i < logp.shape[-1]:
                lp = float(logp[wid_i])
                # Rank = number of entries strictly greater than this one
                # in the full distribution; cheap with numpy on vocab_size.
                rank = int((logp > lp).sum())
                step.watched[wid_i] = TokenCandidate(
                    token_id=wid_i,
                    text=self.piece(wid_i),
                    logprob=lp,
                    rank=rank,
                    is_special=self._is_special(wid_i),
                )
            else:
                # Out-of-vocab id: synthesize an unknown-prob candidate
                # so the renderer can still show a row for it. Same
                # contract as :meth:`Backend.lookup_watch`.
                step.watched[wid_i] = TokenCandidate(
                    token_id=wid_i,
                    text=self.piece(wid_i),
                    logprob=math.nan,
                    rank=-1,
                )
        return step

    def score_prompt(
        self, prompt: str, top_k: int, watch_ids: list[int] | None = None
    ) -> list[StepResult]:
        """Whole-context inspection including the trailing "next" prediction.

        For an N-token prompt this returns N StepResults. The first N-1 rows
        score each prompt token against the actual next one; the final row
        is the distribution *after* the last prompt token, with
        ``chosen=None``. The same Llama.scores tensor already holds that
        row, so the only extra work is one more numpy argpartition.
        """
        np = self._numpy
        watch_ids = watch_ids or []
        ids = self.tokenize(prompt)
        if not ids:
            return []
        logp = self._logsoftmax_all(ids)  # [seq, vocab]
        results: list[StepResult] = []
        for i in range(len(ids)):
            dist = logp[i]
            k = max(1, min(top_k, dist.shape[-1]))
            idx_part = np.argpartition(-dist, k - 1)[:k]
            order = np.argsort(-dist[idx_part])
            idx = idx_part[order]
            vals = dist[idx]
            cands = [
                TokenCandidate(
                    int(j),
                    self.piece(int(j)),
                    float(v),
                    rank,
                    is_special=self._is_special(int(j)),
                )
                for rank, (j, v) in enumerate(zip(idx.tolist(), vals.tolist()))
            ]
            chosen = self._exact_candidate(dist, ids[i + 1]) if i + 1 < len(ids) else None
            watched = {wid: self._exact_candidate(dist, wid) for wid in watch_ids}
            results.append(
                StepResult(
                    position=i + 1,
                    candidates=cands,
                    is_full_vocab=True,
                    chosen=chosen,
                    context_text=self.piece(ids[i]),
                    watched=watched,
                )
            )
        return results

    def _exact_candidate(self, dist, token_id: int) -> TokenCandidate:
        lp = float(dist[token_id])
        rank = int((dist > dist[token_id]).sum())
        return TokenCandidate(
            token_id,
            self.piece(token_id),
            lp,
            rank,
            is_special=self._is_special(token_id),
        )

    # -- speculative-decoding hook (mirrors HFBackend.verify_greedy) ------- #
    def verify_greedy(
        self, context_ids: list[int], draft_ids: list[int]
    ) -> tuple[int, TokenCandidate]:
        """One-forward-pass speculative verification.

        Returns ``(accepted, correction_or_bonus)`` exactly like
        ``HFBackend.verify_greedy``, so any speculator pairing this backend
        with a smaller draft works without UI changes.
        """
        np = self._numpy
        if not self._logits_all:
            raise RuntimeError("verify_greedy requires logits_all=True")
        full = list(context_ids) + list(draft_ids)
        logp = self._logsoftmax_all(full)
        base = len(context_ids) - 1
        accepted = 0
        for i in range(len(draft_ids)):
            pos = base + i
            tgt = int(np.argmax(logp[pos]))
            if tgt == draft_ids[i]:
                accepted += 1
            else:
                return accepted, TokenCandidate(
                    tgt,
                    self.piece(tgt),
                    float(logp[pos, tgt]),
                    0,
                    is_special=self._is_special(tgt),
                )
        pos = len(full) - 1
        tgt = int(np.argmax(logp[pos]))
        return accepted, TokenCandidate(
            tgt,
            self.piece(tgt),
            float(logp[pos, tgt]),
            0,
            is_special=self._is_special(tgt),
        )

    def close(self) -> None:
        # llama-cpp-python's Llama holds a C resource; explicit del helps when
        # tests reload the module repeatedly.
        self._llama = None  # type: ignore[assignment]
        self._cached_ids = []


def _log_softmax(arr, np):
    """Numerically stable row-wise log-softmax for a [seq, vocab] ndarray."""
    m = arr.max(axis=-1, keepdims=True)
    shifted = arr - m
    s = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - s


__all__ = ["LlamaCppPyBackend"]
