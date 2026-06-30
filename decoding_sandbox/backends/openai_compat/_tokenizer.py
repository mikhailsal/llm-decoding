"""Tokenizer mixin: lazy local HF tokenizer load + token-surface helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading

    from tokenizers import Tokenizer

    from decoding_sandbox.core.config import ProviderConfig

log = logging.getLogger(__name__)


class _TokenizerMixin:
    # Composite-class attributes set in ``OpenAICompatBackend.__init__``;
    # declared here under TYPE_CHECKING so the mypy run sees a consistent
    # surface for cross-mixin access. The real definitions and lifetimes
    # live in :mod:`decoding_sandbox.backends.openai_compat.backend`.
    if TYPE_CHECKING:
        provider: ProviderConfig
        model: str
        _tokenizer: Tokenizer | None
        _tokenizer_load_attempted: bool
        _tokenizer_load_error: str
        _tokenizer_load_lock: threading.Lock
        _bos_ids: tuple[int, ...]
        _id_to_text: dict[int, str]
        _text_to_id: dict[str, int]
        _BOS_TOKEN_CANDIDATES: tuple[str, ...]
        _INTERN_ID_BASE: int

        def _intern(self, text: str) -> int: ...

    def _ensure_tokenizer(self) -> Tokenizer | None:
        """Lazy-load the HF tokenizer for ``self.model``; cache the result.

        Returns the loaded ``tokenizers.Tokenizer`` instance, or ``None``
        when no tokenizer mapping is configured for this model OR the
        download fails (gated repo without ``HF_TOKEN``, network down,
        404, ...). After the first call the result is cached -- success
        OR failure -- so we never re-attempt the download within the
        lifetime of this backend instance.

        The graceful-failure path is intentional: ``Capabilities`` and
        the basic text-completion calls still work; we just lose the
        token-array prompt mode and the live token preview for this
        particular model. The first-time warning explains why so the
        operator can grant HF access if they want the full UX.
        """
        if self._tokenizer_load_attempted:
            return self._tokenizer
        with self._tokenizer_load_lock:
            if self._tokenizer_load_attempted:
                return self._tokenizer
            try:
                self._tokenizer = self._do_load_tokenizer()
                if self._tokenizer is not None:
                    self._bos_ids = self._discover_bos_ids(self._tokenizer)
            except Exception as exc:
                self._tokenizer_load_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "tokenizer load failed for %s/%s: %s; "
                    "prepend_token_ids and live token preview will be "
                    "disabled for this model. To enable, "
                    "(a) ensure the mapped repo is correct in "
                    "[providers.%s.tokenizers] and (b) set HF_TOKEN "
                    "in your environment with access to the repo.",
                    self.provider.name,
                    self.model,
                    self._tokenizer_load_error,
                    self.provider.name,
                )
                self._tokenizer = None
            self._tokenizer_load_attempted = True
            return self._tokenizer

    def _do_load_tokenizer(self) -> Tokenizer | None:
        """Resolve the HF repo for ``self.model`` and load tokenizer.json.

        Returns ``None`` (rather than raising) when no repo is configured
        for this model -- that's a regular "no local tokenizer here"
        outcome, not an error. Real network/gating failures raise and
        get caught + logged in ``_ensure_tokenizer``.
        """
        repo = (self.provider.tokenizers or {}).get(self.model)
        if not repo:
            return None
        # Imports kept local so chat-only / lmstudio paths that never
        # need a tokenizer don't pay the rust-binding import cost.
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        # ``huggingface_hub`` reads ``HF_TOKEN`` from the environment when
        # ``token=None``, but a project-local ``.env`` (loaded by
        # ``decoding_sandbox.core.config.load_config``) only reaches the
        # current process's environment -- which is enough here. Passing
        # it explicitly via ``token=os.environ.get(...)`` would be
        # equivalent; we leave the default so ``HF_TOKEN`` ALSO unlocks
        # other paths the library may take (cached metadata refresh,
        # offline lookups). Gated repos still fail when the token lacks
        # access to the repo; the surrounding ``_ensure_tokenizer``
        # catches that and degrades gracefully.
        path = hf_hub_download(repo_id=repo, filename="tokenizer.json")
        tok = Tokenizer.from_file(path)
        log.info(
            "loaded HF tokenizer for %s/%s from repo %s (vocab=%d)",
            self.provider.name,
            self.model,
            repo,
            tok.get_vocab_size(),
        )
        return tok

    def _discover_bos_ids(self, tok: Tokenizer) -> tuple[int, ...]:
        """Best-effort BOS discovery from the tokenizer's special tokens.

        We don't have access to a ``tokenizer_config.json``-style
        explicit ``bos_token`` field via the rust ``tokenizers`` API,
        so we walk the added/special-token decoder and match against a
        small known-suffix list (``_BOS_TOKEN_CANDIDATES``). Returns
        empty when nothing matches -- the UI's "fill BOS" helper will
        grey out and the user can still type any id manually.
        """
        try:
            added = tok.get_added_tokens_decoder()
        except Exception:
            return ()
        # Build a {content -> id} index over only the SPECIAL added
        # tokens (regular added tokens like merged-word entries don't
        # belong on the "fill BOS" button).
        specials: dict[str, int] = {}
        for tid, tok_obj in added.items():
            if getattr(tok_obj, "special", False):
                specials[tok_obj.content] = int(tid)
        for cand in self._BOS_TOKEN_CANDIDATES:
            if cand in specials:
                return (specials[cand],)
        return ()

    def tokenize(self, text: str) -> list[int]:
        """Tokenize ``text`` locally when a HF tokenizer is configured.

        Falls back to the single-intern-id stub (the historical behaviour
        before per-model tokenizer mapping landed) when no tokenizer is
        available -- e.g. lmstudio (model id is just ``"local-model"``,
        no public HF repo) or a Fireworks model whose ``tokenizer.json``
        we couldn't fetch. The stub still satisfies the callers that
        only need a *handle* per text fragment (e.g. the watch-ids
        text-to-id mapping); only the token-array prompt mode and live
        preview features require the real tokenizer.
        """
        tok = self._ensure_tokenizer()
        if tok is None:
            return [self._intern(text)]
        return list(tok.encode(text, add_special_tokens=False).ids)

    def detokenize(self, token_ids: list[int]) -> str:
        tok = self._ensure_tokenizer()
        if tok is None:
            return "".join(self._id_to_text.get(t, "") for t in token_ids)
        # ``skip_special_tokens=False`` so the BOS / EOS the user
        # explicitly typed in the prepend chip-input round-trip back
        # through the preview as their literal text instead of being
        # silently dropped.
        return tok.decode([int(t) for t in token_ids], skip_special_tokens=False)

    def _surface_text(self, token_id: int | None, provider_text: str) -> str:
        """Resolve the renderable surface form of an echoed token.

        Providers detokenize SPECIAL tokens (BOS / EOS / chat markers) to
        an EMPTY string -- Fireworks echoes a prepended
        ``<\uff5cbegin\u2581of\u2581sentence\uff5c>`` back as ``""``, which the
        UI then renders as the dim ``<empty>``
        placeholder. That's inconsistent with the live token preview,
        which routes the same id through :meth:`piece` and shows the real
        token name. When the provider's text is empty but we hold a REAL
        model token id (below the synthetic ``_INTERN_ID_BASE``), fall
        back to the local tokenizer's piece so both views agree. No-ops
        gracefully when there's no local tokenizer (``piece`` returns the
        empty ``_id_to_text`` lookup) or the id is a synthetic intern id.
        """
        if provider_text:
            return provider_text
        if token_id is None or int(token_id) >= self._INTERN_ID_BASE:
            return provider_text
        return self.piece(int(token_id))

    def special_tokens(self) -> list[tuple[int, str]]:
        """Special / added tokens from the mapped HF ``tokenizer.json``.

        Source is ``Tokenizer.get_added_tokens_decoder()`` ({id: AddedToken});
        we keep the entries flagged ``special`` and return ``(id, content)``
        sorted by id. The content string is exactly what ``encode`` matches
        back to the single id, so the Decode workbench can drop it into the
        prompt and trust the round-trip. No tokenizer (chat-only / unmapped
        model) -> empty list (no palette).
        """
        tok = self._ensure_tokenizer()
        if tok is None:
            return []
        out: list[tuple[int, str]] = []
        try:
            decoder = tok.get_added_tokens_decoder()
            for tid, added in decoder.items():
                if getattr(added, "special", False):
                    out.append((int(tid), str(added.content)))
        except Exception:
            return []
        out.sort(key=lambda pair: pair[0])
        return out

    def piece(self, token_id: int) -> str:
        tok = self._ensure_tokenizer()
        if tok is None:
            return self._id_to_text.get(token_id, "")
        # ``id_to_token`` returns the raw vocab string (BPE pieces still
        # carry the GPT-2 ``Ġ`` for word-initial space etc.). Decode of
        # a single id gives the printable surface form, which is what
        # the UI's "piece" RPC consumers expect.
        try:
            return tok.decode([int(token_id)], skip_special_tokens=False)
        except Exception:
            return self._id_to_text.get(token_id, "")
