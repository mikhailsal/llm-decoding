"""Tests for the in-process llama-cpp-python backend.

We don't actually load a GGUF or touch CUDA here -- a tiny ``FakeLlama`` class
plays the role of ``llama_cpp.Llama`` and we install it into ``sys.modules``
before the backend imports it. That exercises every code path
(tokenize/detokenize, full-vocab forward pass, log-softmax math, KV-cache
reuse, score_prompt, verify_greedy) deterministically and without GPU.
"""

from __future__ import annotations

import math
import sys
import types

import pytest


# --------------------------------------------------------------------------- #
# Minimal fake of the llama_cpp surface our backend touches.
# --------------------------------------------------------------------------- #
class _FakeLlama:
    """In-memory ``Llama`` stand-in.

    Tokenizer is byte-level identity (``ord(ch)`` per char). Logits are
    deterministic but distinct per (position, token) so the resulting
    log-softmax has a clear ranking we can assert on.
    """

    def __init__(
        self,
        *,
        model_path: str,
        n_gpu_layers: int,
        n_ctx: int,
        logits_all: bool,
        verbose: bool,
        **_: object,
    ) -> None:
        import numpy as np

        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self.logits_all = logits_all
        # Vocab needs to cover real character codes used in tests ("abc"... "xyz"
        # plus a couple of watch IDs), so make it large enough that token IDs
        # land in-range. Keep it modest so log-softmax tests stay readable.
        self._vocab = 256
        self._scores = np.zeros((n_ctx, self._vocab), dtype=np.float32)
        self._evaluated = 0
        self.eval_calls = 0
        self.reset_calls = 0

    def n_vocab(self) -> int:
        return self._vocab

    def tokenize(self, data: bytes, add_bos: bool = True) -> list[int]:
        return [b for b in data]

    def detokenize(self, ids: list[int]) -> bytes:
        return bytes(int(i) & 0xFF for i in ids)

    def reset(self) -> None:
        self.reset_calls += 1
        self._evaluated = 0
        self._scores[:] = 0.0

    def eval(self, tokens: list[int]) -> None:
        self.eval_calls += 1
        # Synthetic logits: at row r the largest entry sits at column
        # (r * 3 + 1) % vocab. The runner-up at (r * 3 + 2) % vocab. This
        # gives a stable top-1 per position that's easy to predict in tests.
        for offset, tok in enumerate(tokens):
            row = self._evaluated + offset
            # All zeros except a couple of peaks.
            for c in range(self._vocab):
                # base small noise scaled by (c+1) so columns are distinct
                self._scores[row, c] = -float(c) * 0.01
            top = (row * 3 + 1) % self._vocab
            runner = (row * 3 + 2) % self._vocab
            self._scores[row, top] = 5.0
            self._scores[row, runner] = 2.0
            # Make `tok` (the conditioned-on token) get a small positive bump
            # so chosen-token rankings are testable.
            self._scores[row, tok % self._vocab] += 0.5
        self._evaluated += len(tokens)

    @property
    def scores(self):
        return self._scores


def _install_fake_llama_cpp(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLlama]:
    mod = types.ModuleType("llama_cpp")
    mod.Llama = _FakeLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    return _FakeLlama


def _backend(monkeypatch, tmp_path, *, logits_all=True):
    _install_fake_llama_cpp(monkeypatch)
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"")
    from decoding_sandbox.backends.llamacpp_py import LlamaCppPyBackend

    return LlamaCppPyBackend(
        model_path=str(fake_gguf),
        n_gpu_layers=20,
        n_ctx=64,
        logits_all=logits_all,
        verbose=False,
    )


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_discover_model_path_uses_explicit_path(monkeypatch, tmp_path) -> None:
    from decoding_sandbox.backends import llamacpp_py as mod

    target = tmp_path / "explicit.gguf"
    target.write_bytes(b"")
    out = mod._discover_model_path(str(target), [], "**/*.gguf")
    assert out == str(target)


def test_discover_model_path_raises_when_explicit_missing(tmp_path) -> None:
    from decoding_sandbox.backends import llamacpp_py as mod

    with pytest.raises(FileNotFoundError):
        mod._discover_model_path(str(tmp_path / "nope.gguf"), [], "**/*.gguf")


def test_discover_model_path_globs_under_search_dirs(monkeypatch, tmp_path) -> None:
    from decoding_sandbox.backends import llamacpp_py as mod

    nested = tmp_path / "hub" / "models--x" / "snap"
    nested.mkdir(parents=True)
    found = nested / "Qwen3.5-9B-Base-Q4_K_M.gguf"
    found.write_bytes(b"")

    out = mod._discover_model_path(
        None, [str(tmp_path)], "**/Qwen3.5-9B-Base-Q4_K_M.gguf"
    )
    assert out == str(found)


def test_discover_model_path_raises_with_helpful_message_when_nothing_matches(
    tmp_path,
) -> None:
    from decoding_sandbox.backends import llamacpp_py as mod

    with pytest.raises(FileNotFoundError, match="No GGUF matching"):
        mod._discover_model_path(None, [str(tmp_path)], "**/*.gguf")


def test_discover_model_path_ignores_missing_search_dirs(tmp_path) -> None:
    from decoding_sandbox.backends import llamacpp_py as mod

    real = tmp_path / "a"
    real.mkdir()
    (real / "m.gguf").write_bytes(b"")
    out = mod._discover_model_path(
        None, ["/does/not/exist", str(tmp_path)], "**/*.gguf"
    )
    assert out.endswith("m.gguf")


# --------------------------------------------------------------------------- #
# Capabilities + tokenization
# --------------------------------------------------------------------------- #
def test_capabilities_advertise_full_vocab_when_logits_all_true(
    monkeypatch, tmp_path
) -> None:
    b = _backend(monkeypatch, tmp_path)
    caps = b.capabilities

    assert caps.full_vocab is True
    assert caps.prompt_logprobs is True
    assert caps.max_top_logprobs == 256
    assert caps.can_force_token is True
    assert "logits_all=True" in caps.notes


def test_capabilities_downgrade_when_logits_all_false(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path, logits_all=False)
    caps = b.capabilities
    assert caps.full_vocab is False
    assert caps.prompt_logprobs is False
    assert "last-position only" in caps.notes


def test_tokenize_round_trips_through_detokenize(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path)
    ids = b.tokenize("abc")

    assert ids == [97, 98, 99]
    assert b.detokenize(ids) == "abc"


def test_piece_caches_individual_token_detokenization(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path)
    a1 = b.piece(65)
    a2 = b.piece(65)
    assert a1 == a2 == "A"
    # Cache size grows with distinct ids only.
    b.piece(66)
    assert set(b._piece_cache) == {65, 66}


# --------------------------------------------------------------------------- #
# Forward pass + log-softmax sanity
# --------------------------------------------------------------------------- #
def test_next_distribution_returns_ranked_full_vocab_candidates(
    monkeypatch, tmp_path
) -> None:
    b = _backend(monkeypatch, tmp_path)
    step = b.next_distribution(b.tokenize("abc"), top_k=4)

    assert step.is_full_vocab is True
    assert step.position == 3
    # Top-1 column at row r is (r * 3 + 1) % vocab. Last row index = 2 -> 7.
    assert step.candidates[0].token_id == 7
    # Rank-2 column = (2 * 3 + 2) % vocab = 8.
    assert step.candidates[1].token_id == 8
    # Probabilities must sum to <=1 across any top-k slice; full vocab sums to 1.
    total = sum(math.exp(c.logprob) for c in step.candidates)
    assert 0.0 < total <= 1.0001


def test_next_distribution_clamps_top_k_to_vocab_size(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path)
    step = b.next_distribution(b.tokenize("ab"), top_k=10_000)

    assert len(step.candidates) == 256  # vocab size
    assert step.candidates[0].rank == 0


def test_score_prompt_records_actual_token_at_each_position(
    monkeypatch, tmp_path
) -> None:
    b = _backend(monkeypatch, tmp_path)
    steps = b.score_prompt("abc", top_k=4, watch_ids=[5, 9])

    # Three positions in "abc" but score_prompt skips the first (no preceding
    # context to condition on) -> two steps.
    assert len(steps) == 2
    for st in steps:
        assert st.is_full_vocab is True
        assert st.chosen is not None
        assert st.chosen.rank >= 0  # exact rank, never -1 for full vocab
        assert set(st.watched) == {5, 9}
        for w in st.watched.values():
            assert not math.isnan(w.logprob)


def test_score_prompt_returns_empty_for_single_token_prompt(
    monkeypatch, tmp_path
) -> None:
    b = _backend(monkeypatch, tmp_path)
    assert b.score_prompt("a", top_k=4) == []


def test_score_prompt_raises_when_logits_all_false(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path, logits_all=False)
    with pytest.raises(RuntimeError, match="logits_all=False"):
        b.score_prompt("abc", top_k=4)


def test_log_softmax_helper_normalizes_rows() -> None:
    import numpy as np
    from decoding_sandbox.backends.llamacpp_py import _log_softmax

    arr = np.array([[1.0, 2.0, 3.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    out = _log_softmax(arr, np)

    sums = np.exp(out).sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# KV-cache reuse
# --------------------------------------------------------------------------- #
def test_score_prompt_calls_eval_only_once_for_a_fresh_prompt(
    monkeypatch, tmp_path
) -> None:
    b = _backend(monkeypatch, tmp_path)
    fake: _FakeLlama = b._llama  # type: ignore[assignment]

    b.score_prompt("abcdef", top_k=4)

    assert fake.eval_calls == 1
    assert fake.reset_calls == 0  # nothing cached before -> no reset


def test_extending_context_reuses_cache_no_reset(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path)
    fake: _FakeLlama = b._llama  # type: ignore[assignment]

    b.next_distribution(b.tokenize("ab"), top_k=2)
    eval_after_first = fake.eval_calls
    reset_after_first = fake.reset_calls

    b.next_distribution(b.tokenize("abcd"), top_k=2)

    # Strict extension -> one extra eval, no reset.
    assert fake.eval_calls == eval_after_first + 1
    assert fake.reset_calls == reset_after_first


def test_non_extension_context_resets_cache(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path)
    fake: _FakeLlama = b._llama  # type: ignore[assignment]

    b.next_distribution(b.tokenize("abcd"), top_k=2)
    b.next_distribution(b.tokenize("xyz"), top_k=2)

    assert fake.reset_calls >= 1


def test_next_distribution_with_empty_ids_returns_empty_step(
    monkeypatch, tmp_path
) -> None:
    b = _backend(monkeypatch, tmp_path)
    step = b.next_distribution([], top_k=4)
    assert step.candidates == []
    assert step.position == 0


# --------------------------------------------------------------------------- #
# Speculative-decoding hook
# --------------------------------------------------------------------------- #
def test_verify_greedy_accepts_matching_drafts_and_emits_bonus(
    monkeypatch, tmp_path
) -> None:
    b = _backend(monkeypatch, tmp_path)
    ctx = b.tokenize("ab")
    # Use the model's own greedy choices as the drafts -> all accepted.
    drafts = []
    for _ in range(3):
        step = b.next_distribution(ctx + drafts, top_k=1)
        drafts.append(step.candidates[0].token_id)

    accepted, correction = b.verify_greedy(ctx, drafts)

    assert accepted == len(drafts)
    assert correction is not None
    assert correction.text != ""


def test_verify_greedy_rejects_wrong_draft_and_returns_correction(
    monkeypatch, tmp_path
) -> None:
    b = _backend(monkeypatch, tmp_path)
    ctx = b.tokenize("ab")
    # Deliberately wrong draft id 0. verify_greedy reads the row at
    # ``base = len(ctx) - 1 = 1`` (predicting what follows ctx[1]) -- the
    # fake's top column at row 1 is (1*3+1) % 256 = 4.
    accepted, correction = b.verify_greedy(ctx, [0])
    assert accepted == 0
    assert correction.rank == 0
    assert correction.token_id == 4


def test_verify_greedy_raises_when_logits_all_false(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path, logits_all=False)
    with pytest.raises(RuntimeError, match="logits_all=True"):
        b.verify_greedy(b.tokenize("ab"), [1])


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
def test_close_releases_inner_llama(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path)
    b.close()
    assert b._llama is None
    assert b._cached_ids == []
