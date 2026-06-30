"""Parsing mixin: token-id interning and provider-response candidate parsing."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from dsbx.core.backend import candidates_from_logprobs
from dsbx.core.engine import GenStep
from dsbx.core.samplers import SamplerDecision
from dsbx.core.types import StepResult, TokenCandidate

log = logging.getLogger(__name__)


class _ParsingMixin:
    # Composite-class attributes set in ``OpenAICompatBackend.__init__``;
    # declared here under TYPE_CHECKING so mypy sees the surface this
    # mixin reaches into (``_id_to_text`` / ``_text_to_id`` / ``_intern``
    # live across mixins, ``_provider_flag`` / ``_surface_text`` /
    # ``lookup_watch`` come from sibling mixins). Runtime semantics are
    # unchanged -- all real definitions live in
    # :mod:`dsbx.backends.openai_compat.backend` and
    # ``_tokenizer.py``.
    if TYPE_CHECKING:
        _id_to_text: dict[int, str]
        _text_to_id: dict[str, int]
        _INTERN_ID_BASE: int

        def _provider_flag(self, name: str) -> bool: ...
        def _surface_text(self, token_id: int | None, provider_text: str) -> str: ...
        def lookup_watch(self, step: StepResult, token_id: int) -> TokenCandidate: ...

    def _intern(self, text: str) -> int:
        if text not in self._text_to_id:
            tid = self._INTERN_ID_BASE + len(self._text_to_id)
            self._text_to_id[text] = tid
            self._id_to_text[tid] = text
        return self._text_to_id[text]

    # -- HF tokenizer ----------------------------------------------------- #
    # Common heuristics for picking the "BOS-ish" id out of a tokenizer
    # that doesn't expose a dedicated ``bos_token`` field. Listed in
    # priority order: we walk this list and take the first matching
    # added/special token. Order matters -- some models reuse
    # ``<|endoftext|>`` as their BOS (Qwen Base), so we have to prefer
    # an explicit start-of-text marker when both are present.
    _BOS_TOKEN_CANDIDATES: tuple[str, ...] = (
        "<|startoftext|>",  # gpt-oss family
        "<|begin_of_text|>",  # Llama 3.x
        "<s>",  # Llama 2, Mistral
        # DeepSeek uses U+FF5C FULLWIDTH VERTICAL LINE (｜) instead of the  # noqa: RUF003
        # ASCII pipe -- the same string with regular ``|`` won't match.
        # Two variants exist in the wild: the original (V2/V3) form
        # spells "begin_of_sentence" with a U+2581 LOWER ONE EIGHTH BLOCK
        # (▁) between words, and the V3.1+ form uses underscores. Try
        # both so the discovery survives a model upgrade.
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
        "<\uff5cbegin_of_sentence\uff5c>",
        "<|im_start|>",  # Qwen / ChatML chat marker (best-effort)
        "]~!b[",  # MiniMax M2.x declared bos_token (literal)
        "<|endoftext|>",  # GPT-2, Qwen Base (fallback only)
    )

    def _cands_from_dict(self, d: dict[str, float]) -> list[TokenCandidate]:
        triples = [(self._intern(tok), tok, float(lp)) for tok, lp in d.items()]
        return candidates_from_logprobs(triples)

    def _cands_from_list(self, items: list[dict]) -> list[TokenCandidate]:
        triples = [(self._intern(i["token"]), i["token"], float(i["logprob"])) for i in items]
        return candidates_from_logprobs(triples)

    def _cands_from_new_logprobs(self, items: list[dict]) -> list[TokenCandidate]:
        """Parse a NewLogProbs ``top_logprobs[i]`` array into candidates.

        NewLogProbs (Fireworks) carries real model token ids, the token
        text, logprob, and (for the chosen position only) a
        ``sampling_logprob`` representing the post-filter probability.
        Items are ordered by descending probability per the server, so
        we can take the input list's index as the candidate rank
        without re-sorting.

        Crucially: we use the REAL ``token_id`` instead of calling
        :meth:`_intern`. That's the whole reason we ship
        ``logprobs: true`` -- so ``--watch-id`` references map to the
        same id the model actually emitted, and so the per-token id
        list on each ``GenStep.tokens_before`` is meaningful instead
        of being a stream of synthetic intern hashes. We do still
        populate ``_id_to_text`` for ``piece`` / ``detokenize`` so the
        TUI/web renderer keeps working unchanged.
        """
        out: list[TokenCandidate] = []
        for rank, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            tid_raw = item.get("token_id")
            if tid_raw is None:
                # Some upstreams omit token_id for the runner-ups but
                # include the text. Fall back to intern so the
                # candidate still appears in the table; the warning
                # only fires once per session via the dedup'd cache.
                text = str(item.get("token", ""))
                tid = self._intern(text)
            else:
                tid = int(tid_raw)
                text = str(item.get("token", ""))
                if text:
                    self._id_to_text.setdefault(tid, text)
                else:
                    # Special tokens echo back blank; show the local
                    # tokenizer's piece so a BOS/EOS alt isn't a void cell.
                    text = self._surface_text(tid, text)
            lp = item.get("logprob")
            logprob = float(lp) if lp is not None else float("nan")
            out.append(
                TokenCandidate(
                    token_id=tid,
                    text=text,
                    logprob=logprob,
                    rank=rank,
                )
            )
        return out

    def _genstep_from_emit_record(
        self,
        record: tuple[int | None, str, float, Any, int | None],
        *,
        tokens_before: list[int],
        step_idx: int,
        is_last: bool,
        last_finish_reason: str | None,
        watch_ids: Sequence[int],
        note: str,
    ) -> GenStep:
        """Build one emitted-token :class:`GenStep` from a streamed record.

        Shared by :meth:`stream_native` and
        :meth:`stream_native_with_echo` so both incremental paths produce
        byte-identical GenSteps. ``tokens_before`` is mutated in place
        (the emitted id is appended) so successive calls advance the
        context exactly as the old buffered loops did. ``is_last`` stamps
        the provider's ``finish_reason``-derived ``stop_reason`` onto the
        terminal step; non-terminal steps carry ``stop_reason=None``.
        """
        tok_id_real, tok_text, tok_lp, top_entry, smc = record
        if isinstance(top_entry, list) and self._provider_flag("supports_new_logprobs"):
            cands = self._cands_from_new_logprobs(top_entry)
        else:
            cands = self._candidates_from_top_entry(top_entry)
        if tok_id_real is not None:
            tok_id = tok_id_real
            if tok_text and tok_id not in self._id_to_text:
                self._id_to_text[tok_id] = tok_text
        else:
            tok_id = self._intern(tok_text)
        # Specials (a prepended BOS in the echo prefix, or a generated
        # EOS / chat marker) echo back blank -> render the piece so the
        # running-completion prefix and the SEED row show the real token
        # name instead of the dim ``<empty>``, matching the live token
        # preview. No-op for non-empty text and for synthetic intern ids.
        text = self._surface_text(tok_id, tok_text)
        if smc is not None:
            for c in cands:
                c.sampling_mask_count = smc
        chosen = next((c for c in cands if c.token_id == tok_id), None)
        if chosen is None:
            # The emitted token didn't make the top_k cut. Synthesize
            # a candidate with rank=-1 so the UI still has something
            # to show; this matches what ``score_prompt`` does.
            chosen = TokenCandidate(tok_id, text, tok_lp, rank=-1, sampling_mask_count=smc)
        greedy_id = cands[0].token_id if cands else tok_id
        sr = StepResult(
            position=len(tokens_before),
            candidates=cands,
            is_full_vocab=False,
            chosen=chosen,
        )
        # Per-step watch column: fish each watched id out of the
        # same chunk's top_k. Cloud providers cap top_k at 5
        # (Fireworks) / 20 (NIM/OpenRouter), so ids outside that
        # window render as ``rank=-1, logprob=NaN`` (dim "—" in
        # the UI).
        for wid in watch_ids:
            sr.watched[int(wid)] = self.lookup_watch(sr, int(wid))
        stop_reason = self._finish_reason_to_stop(last_finish_reason) if is_last else None
        decision = SamplerDecision(
            token_id=tok_id,
            token_text=text,
            kept=[],
            greedy_token_id=greedy_id,
            note=note,
        )
        gs = GenStep(
            step=step_idx,
            tokens_before=list(tokens_before),
            step_result=sr,
            decision=decision,
            stop_reason=stop_reason,
        )
        tokens_before.append(tok_id)
        return gs

    def _stepresult_from_echo_record(
        self,
        record: tuple[int | None, str, float, Any, int | None, int | None, bool],
        *,
        pos_idx: int,
        watch_ids: Sequence[int],
    ) -> StepResult:
        """Build one prompt-echo :class:`StepResult` from a streamed record.

        Extracted from the old buffered echo loop so
        :meth:`stream_native_with_echo` can yield each echoed prompt
        position the moment it's classified instead of after the whole
        stream is drained.
        """
        tok_id_real, tok_text, tok_lp, top_entry, smc, _text_off, _has_signal = record
        if isinstance(top_entry, list) and self._provider_flag("supports_new_logprobs"):
            cands = self._cands_from_new_logprobs(top_entry)
        else:
            cands = self._candidates_from_top_entry(top_entry)
        if tok_id_real is not None:
            tok_id = tok_id_real
            if tok_text and tok_id not in self._id_to_text:
                self._id_to_text[tok_id] = tok_text
        else:
            tok_id = self._intern(tok_text)
        text = self._surface_text(tok_id, tok_text)
        if smc is not None:
            for c in cands:
                c.sampling_mask_count = smc
        chosen = next((c for c in cands if c.token_id == tok_id), None)
        if chosen is None:
            # Fireworks (and OpenAI-compat echo in general) emits
            # position 0 with NO ``top_logprobs`` -- the model has no
            # prior context to score against, so the upstream returns
            # ``logprob: 0.0`` as a placeholder rather than a real value.
            # Propagating 0.0 would render the token at exp(0.0)=100%,
            # a lie: autoregressive models can't predict position 0
            # without BOS conditioning. Detect the placeholder by the
            # absence of ``cands`` and downgrade to NaN so the UI shows
            # an honest "?" instead. Emit positions outside top-K keep
            # their real ``tok_lp`` (cands populated), so this guard
            # doesn't touch them.
            effective_lp = float("nan") if not cands else tok_lp
            chosen = TokenCandidate(tok_id, text, effective_lp, rank=-1, sampling_mask_count=smc)
        prompt_step = StepResult(
            position=pos_idx,
            candidates=cands,
            is_full_vocab=False,
            chosen=chosen,
            context_text=text,
        )
        # Watch column on echoed prompt positions: same top-k-only
        # contract as the per-emitted-step path.
        for wid in watch_ids:
            prompt_step.watched[int(wid)] = self.lookup_watch(prompt_step, int(wid))
        return prompt_step

    def _candidates_from_top_entry(self, top_entry: Any) -> list[TokenCandidate]:
        """Adapt either /completions (dict) or /chat (list) per-token shapes.

        Fireworks /completions returns ``top_logprobs[i]`` as a dict
        ``{token_text: logprob}``; the chat schema returns a list of
        ``{token, logprob}`` records. Both flow through here so future
        chat-streaming support (NIM, OpenRouter) reuses the parser.
        """
        if top_entry is None:
            return []
        if isinstance(top_entry, dict):
            return self._cands_from_dict(top_entry)
        if isinstance(top_entry, list):
            return self._cands_from_list(top_entry)
        return []

    @staticmethod
    def _finish_reason_to_stop(reason: str | None) -> str | None:
        """Map provider ``finish_reason`` to our ``GenStep.stop_reason``.

        ``"length"`` always means we ran out of ``max_tokens``;
        ``"stop"`` means either a stop sequence matched OR the model
        emitted its native EOS. We can't distinguish those two cases
        from the wire (the provider collapses them), so we report
        ``"user_stop"`` -- the engine's CLI footer happens to also say
        "stopped on EOS" only when it knows the EOS id, so this is the
        honest reading.
        """
        if reason == "length":
            return "max_tokens"
        if reason == "stop":
            return "user_stop"
        return None

    # -- model discovery -------------------------------------------------- #
