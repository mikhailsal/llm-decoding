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

    Class attribute ``EOS_ID`` controls what ``token_eos()`` returns; tests
    can override it (e.g. ``cls.EOS_ID = -1`` to simulate "no EOS exposed")
    by subclassing or by monkeypatching before construction.
    """

    EOS_ID = 250
    EOT_ID: int | None = None  # not all bindings expose token_eot()
    # BOS plumbing: ``BOS_ID`` is what ``token_bos()`` returns;
    # ``METADATA`` lets a test simulate GGUF metadata flags (a real
    # llama.cpp model exposes this dict for tokenizer.ggml.* keys).
    BOS_ID: int = 1
    METADATA: dict[str, str] = {}

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
        # token_eot is only set when the class advertises one; this matches
        # what newer llama-cpp-python bindings do for chat templates.
        if self.EOT_ID is not None:
            self.token_eot = lambda: self.EOT_ID  # type: ignore[assignment]
        # ``metadata`` is a plain dict on real llama-cpp-python; copy
        # the class default so tests can mutate per-instance without
        # bleeding across runs.
        self.metadata = dict(self.METADATA)

    def token_bos(self) -> int:
        return self.BOS_ID

    def token_eos(self) -> int:
        return self.EOS_ID

    def n_vocab(self) -> int:
        return self._vocab

    def tokenize(self, data: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        return list(data)

    def detokenize(self, ids: list[int], special: bool = False) -> bytes:
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

    out = mod._discover_model_path(None, [str(tmp_path)], "**/Qwen3.5-9B-Base-Q4_K_M.gguf")
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
    out = mod._discover_model_path(None, ["/does/not/exist", str(tmp_path)], "**/*.gguf")
    assert out.endswith("m.gguf")


# --------------------------------------------------------------------------- #
# Capabilities + tokenization
# --------------------------------------------------------------------------- #
def test_capabilities_advertise_full_vocab_when_logits_all_true(monkeypatch, tmp_path) -> None:
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
def test_next_distribution_returns_ranked_full_vocab_candidates(monkeypatch, tmp_path) -> None:
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


def test_score_prompt_records_actual_token_at_each_position(monkeypatch, tmp_path) -> None:
    """For an N-token prompt the new contract is N steps:
    N-1 scored rows plus a trailing "predict next" row with chosen=None.
    """
    b = _backend(monkeypatch, tmp_path)
    steps = b.score_prompt("abc", top_k=4, watch_ids=[5, 9])

    assert len(steps) == 3  # was 2 before the trailing-prediction fix
    scored, trailing = steps[:-1], steps[-1]
    for st in scored:
        assert st.is_full_vocab is True
        assert st.chosen is not None
        assert st.chosen.rank >= 0  # exact rank, never -1 for full vocab
        assert set(st.watched) == {5, 9}
        for w in st.watched.values():
            assert not math.isnan(w.logprob)
    assert trailing.chosen is None
    assert trailing.position == 3  # one past the last token (1-indexed)
    assert set(trailing.watched) == {5, 9}
    for w in trailing.watched.values():
        # Exact, full-vocab read -- the whole point of the fix is that
        # P(EOS) after the prompt finally has a real value.
        assert not math.isnan(w.logprob)


def test_score_prompt_includes_trailing_step_for_single_token_prompt(monkeypatch, tmp_path) -> None:
    """A 1-token prompt has no scored rows (nothing to score against) but
    still has a trailing prediction. Empty list would silently drop the
    only interesting position for a 1-token prompt."""
    b = _backend(monkeypatch, tmp_path)
    [trailing] = b.score_prompt("a", top_k=4, watch_ids=[5])
    assert trailing.chosen is None
    assert trailing.position == 1
    assert not math.isnan(trailing.watched[5].logprob)


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
def test_score_prompt_calls_eval_only_once_for_a_fresh_prompt(monkeypatch, tmp_path) -> None:
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


def test_next_distribution_with_empty_ids_returns_empty_step(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path)
    step = b.next_distribution([], top_k=4)
    assert step.candidates == []
    assert step.position == 0


# --------------------------------------------------------------------------- #
# Speculative-decoding hook
# --------------------------------------------------------------------------- #
def test_verify_greedy_accepts_matching_drafts_and_emits_bonus(monkeypatch, tmp_path) -> None:
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


def test_verify_greedy_rejects_wrong_draft_and_returns_correction(monkeypatch, tmp_path) -> None:
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


# --------------------------------------------------------------------------- #
# EOS discovery + is_special
# --------------------------------------------------------------------------- #
def test_capabilities_expose_eos_from_token_eos(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path)
    assert b.capabilities.eos_token_ids == (250,)


def test_capabilities_include_token_eot_when_binding_exposes_it(monkeypatch, tmp_path) -> None:
    """Newer llama-cpp-python builds expose ``Llama.token_eot()`` for chat
    templates. The backend should pick that up too."""

    class _LlamaWithEot(_FakeLlama):
        EOT_ID = 251

    mod = types.ModuleType("llama_cpp")
    mod.Llama = _LlamaWithEot
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"")
    from decoding_sandbox.backends.llamacpp_py import LlamaCppPyBackend

    b = LlamaCppPyBackend(
        model_path=str(fake_gguf),
        n_gpu_layers=20,
        n_ctx=64,
        logits_all=True,
        verbose=False,
    )
    assert b.capabilities.eos_token_ids == (250, 251)


def test_negative_eos_id_is_dropped(monkeypatch, tmp_path) -> None:
    """A negative id from llama.cpp means "no EOS configured" -- it shouldn't
    end up in capabilities (otherwise generate() would never stop)."""

    class _NoEos(_FakeLlama):
        EOS_ID = -1

    mod = types.ModuleType("llama_cpp")
    mod.Llama = _NoEos
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"")
    from decoding_sandbox.backends.llamacpp_py import LlamaCppPyBackend

    b = LlamaCppPyBackend(
        model_path=str(fake_gguf),
        n_gpu_layers=20,
        n_ctx=64,
        logits_all=True,
        verbose=False,
    )
    assert b.capabilities.eos_token_ids == ()


def test_capabilities_drop_bos_when_metadata_says_add_bos_token_false(
    monkeypatch, tmp_path
) -> None:
    """``tokenizer.ggml.add_bos_token=false`` -> ``bos_token_ids == ()``.

    Qwen3.5-9B-Base in particular ships this metadata and has NO
    ``bos_token_id`` declared. ``Llama.token_bos()`` then falls back
    to id 11, which happens to be ``,`` in Qwen's vocab. Without the
    metadata veto we would advertise ``[11]`` as BOS and the workbench
    would helpfully prepend a literal comma to every prompt -- exactly
    the bug the dsbx-host-py "running completion starts with a comma" UX
    pointed at. The fix: trust the model author, return an empty tuple.
    """

    class _NoBosViaMetadata(_FakeLlama):
        BOS_ID = 11
        METADATA = {"tokenizer.ggml.add_bos_token": "false"}

    mod = types.ModuleType("llama_cpp")
    mod.Llama = _NoBosViaMetadata
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"")
    from decoding_sandbox.backends.llamacpp_py import LlamaCppPyBackend

    b = LlamaCppPyBackend(
        model_path=str(fake_gguf),
        n_gpu_layers=20,
        n_ctx=64,
        logits_all=True,
        verbose=False,
    )
    assert b.capabilities.bos_token_ids == ()


def test_capabilities_drop_bos_when_metadata_omits_bos_token_id(monkeypatch, tmp_path) -> None:
    """No ``tokenizer.ggml.bos_token_id`` key -> empty tuple.

    Same shape as above but the model author left the key off entirely
    instead of saying ``add_bos_token=false``. We still want the empty
    tuple: if the model didn't bother declaring a BOS, ``token_bos()``'s
    fallback id is uninterpretable noise.
    """

    class _NoBosId(_FakeLlama):
        BOS_ID = 42  # some arbitrary fallback
        METADATA: dict[str, str] = {}  # no bos_token_id key at all

    mod = types.ModuleType("llama_cpp")
    mod.Llama = _NoBosId
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"")
    from decoding_sandbox.backends.llamacpp_py import LlamaCppPyBackend

    b = LlamaCppPyBackend(
        model_path=str(fake_gguf),
        n_gpu_layers=20,
        n_ctx=64,
        logits_all=True,
        verbose=False,
    )
    assert b.capabilities.bos_token_ids == ()


def test_capabilities_expose_bos_when_metadata_declares_real_one(monkeypatch, tmp_path) -> None:
    """add_bos_token=true + bos_token_id present -> tuple carries the id.

    The happy path: Llama-3 style models declare a real BOS in
    metadata, ``token_bos()`` returns the matching id, and the UI's
    "fill BOS" helper picks it up automatically.
    """

    class _WithBos(_FakeLlama):
        BOS_ID = 128000
        METADATA = {
            "tokenizer.ggml.add_bos_token": "true",
            "tokenizer.ggml.bos_token_id": "128000",
        }

    mod = types.ModuleType("llama_cpp")
    mod.Llama = _WithBos
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"")
    from decoding_sandbox.backends.llamacpp_py import LlamaCppPyBackend

    b = LlamaCppPyBackend(
        model_path=str(fake_gguf),
        n_gpu_layers=20,
        n_ctx=64,
        logits_all=True,
        verbose=False,
    )
    assert b.capabilities.bos_token_ids == (128000,)


def test_is_special_true_for_eos_token(monkeypatch, tmp_path) -> None:
    b = _backend(monkeypatch, tmp_path)
    step = b.next_distribution(b.tokenize("ab"), top_k=256)
    # Find the EOS candidate; it must carry is_special=True.
    eos = next(c for c in step.candidates if c.token_id == 250)
    assert eos.is_special is True


def test_is_special_true_for_braced_tokens(monkeypatch, tmp_path) -> None:
    """Tokens whose detokenized text matches ``<|...|>`` also get is_special."""

    class _LlamaWithBracedToken(_FakeLlama):
        def detokenize(self, ids, special: bool = False):
            # Mark id 7 as a special braced token; everything else is identity.
            if list(ids) == [7]:
                return b"<|im_end|>"
            return super().detokenize(ids, special=special)

    mod = types.ModuleType("llama_cpp")
    mod.Llama = _LlamaWithBracedToken
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"")
    from decoding_sandbox.backends.llamacpp_py import LlamaCppPyBackend

    b = LlamaCppPyBackend(
        model_path=str(fake_gguf),
        n_gpu_layers=20,
        n_ctx=64,
        logits_all=True,
        verbose=False,
    )
    step = b.next_distribution(b.tokenize("ab"), top_k=256)
    braced = next(c for c in step.candidates if c.token_id == 7)
    assert braced.is_special is True
    # And id 7 specifically -- not every candidate -- is flagged.
    others = [c.is_special for c in step.candidates if c.token_id != 7 and c.token_id != 250]
    assert all(s is False for s in others)


def test_piece_renders_special_token_name(monkeypatch, tmp_path) -> None:
    """``piece`` must surface a control token's name, never an empty string.

    Real llama.cpp detokenizes control tokens to ``""`` UNLESS special=True;
    the backend passes special=True so EOS/BOS read as ``<|endoftext|>`` in
    the UI instead of the misleading dim ``<empty>``.
    """

    class _LlamaSpecialAware(_FakeLlama):
        def detokenize(self, ids, special: bool = False):
            if list(ids) == [42]:
                # Control token: blank without special, named with it.
                return b"<|endoftext|>" if special else b""
            return super().detokenize(ids, special=special)

    mod = types.ModuleType("llama_cpp")
    mod.Llama = _LlamaSpecialAware
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"")
    from decoding_sandbox.backends.llamacpp_py import LlamaCppPyBackend

    b = LlamaCppPyBackend(model_path=str(fake_gguf), n_gpu_layers=20, n_ctx=64, logits_all=True)
    assert b.piece(42) == "<|endoftext|>"


def test_special_tokens_scans_control_and_user_defined(monkeypatch, tmp_path) -> None:
    """``special_tokens`` enumerates CONTROL / USER_DEFINED ids via attr API."""

    class _Model:
        model = object()

    class _LlamaWithVocab(_FakeLlama):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._model = _Model()

        def detokenize(self, ids, special: bool = False):
            names = {5: b"<|im_start|>", 9: b"<|endoftext|>"}
            if special and list(ids) and next(iter(ids)) in names:
                return names[next(iter(ids))]
            return super().detokenize(ids, special=special)

    mod = types.ModuleType("llama_cpp")
    mod.Llama = _LlamaWithVocab
    mod.llama_model_get_vocab = lambda m: "VOCAB"
    # id 5 = CONTROL (1<<3), id 9 = USER_DEFINED (1<<4), rest = normal.
    attr_map = {5: 1 << 3, 9: 1 << 4}
    mod.llama_vocab_get_attr = lambda vocab, i: attr_map.get(i, 0)
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"")
    from decoding_sandbox.backends.llamacpp_py import LlamaCppPyBackend

    b = LlamaCppPyBackend(model_path=str(fake_gguf), n_gpu_layers=20, n_ctx=64, logits_all=True)
    specials = b.special_tokens()
    assert (5, "<|im_start|>") in specials
    assert (9, "<|endoftext|>") in specials
    # Caching: a second call returns the same object without re-scanning.
    assert b.special_tokens() is specials


def test_special_tokens_empty_when_attr_api_missing(monkeypatch, tmp_path) -> None:
    """No low-level attr API on this llama_cpp build -> empty list, no crash."""
    mod = types.ModuleType("llama_cpp")
    mod.Llama = _FakeLlama  # no llama_model_get_vocab / llama_vocab_get_attr
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"")
    from decoding_sandbox.backends.llamacpp_py import LlamaCppPyBackend

    b = LlamaCppPyBackend(model_path=str(fake_gguf), n_gpu_layers=20, n_ctx=64, logits_all=True)
    assert b.special_tokens() == []
