"""HuggingFace transformers backend: the full-vocabulary white box.

One forward pass yields logits of shape ``[1, seq, vocab]`` -- the exact
distribution at every position. This is the only backend that returns true
full-vocab probabilities, exact ranks, and whole-context inspection in a single
pass, and it can force arbitrary tokens (for manual/speculative decoding).

torch/transformers are imported lazily so the CLI still runs on machines without
them (e.g. the client); this backend is only instantiated on dsbx-host.
"""

from __future__ import annotations

from collections.abc import Sequence

from decoding_sandbox.core.backend import Backend
from decoding_sandbox.core.types import Capabilities, StepResult, TokenCandidate


class HFBackend(Backend):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-1.7B-Base",
        *,
        fallback_model: str | None = "Qwen/Qwen3-1.7B-Base",
        load_in_4bit: bool = True,
        gpu_mem: str = "4500MiB",
        cpu_mem: str = "13GiB",
    ) -> None:
        import torch  # noqa: F401  (validate availability early)

        self._torch = torch
        self.model_id = model_id
        self._piece_cache: dict[int, str] = {}
        try:
            self._load(model_id, load_in_4bit, gpu_mem, cpu_mem)
            self.loaded_model = model_id
        except Exception as exc:  # noqa: BLE001
            if not fallback_model or fallback_model == model_id:
                raise
            print(
                f"[HFBackend] {model_id} failed ({type(exc).__name__}); "
                f"falling back to {fallback_model}"
            )
            self._load(fallback_model, load_in_4bit, gpu_mem, cpu_mem)
            self.loaded_model = fallback_model

    def _load(self, model_id: str, four_bit: bool, gpu_mem: str, cpu_mem: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        kwargs: dict = dict(device_map="auto", trust_remote_code=True)
        if four_bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            kwargs["max_memory"] = {0: gpu_mem, "cpu": cpu_mem}
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self._eos_ids: tuple[int, ...] = _gather_eos_ids(self.model.config, self.tokenizer)
        self._bos_ids: tuple[int, ...] = _gather_bos_ids(self.model.config, self.tokenizer)
        self._special_ids: frozenset[int] = frozenset(
            int(i) for i in (getattr(self.tokenizer, "all_special_ids", []) or [])
        )

    @property
    def capabilities(self) -> Capabilities:
        vocab = int(getattr(self.model.config, "vocab_size", 0) or len(self.tokenizer))
        return Capabilities(
            name=f"hf:{self.loaded_model}",
            full_vocab=True,
            prompt_logprobs=True,
            max_top_logprobs=vocab,
            can_force_token=True,
            notes="exact full-vocab distribution; whole-context in one forward pass.",
            eos_token_ids=self._eos_ids,
            bos_token_ids=self._bos_ids,
            supports_prepend_token_ids=True,
            # The HF backend always has the real tokenizer in-process,
            # so the live token preview in the Decode workbench is
            # safe and useful here.
            supports_local_tokenize=True,
        )

    def _is_special(self, token_id: int) -> bool:
        return token_id in self._special_ids or token_id in self._eos_ids

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, return_tensors=None)["input_ids"]

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def piece(self, token_id: int) -> str:
        if token_id not in self._piece_cache:
            self._piece_cache[token_id] = self.tokenizer.decode([token_id])
        return self._piece_cache[token_id]

    def special_tokens(self) -> list[tuple[int, str]]:
        """Special / added tokens from the transformers tokenizer.

        Prefers ``added_tokens_decoder`` ({id: AddedToken}) filtered to the
        ``special`` entries; falls back to zipping ``all_special_ids`` with
        ``all_special_tokens`` on older tokenizers that don't expose the
        decoder map. Returns ``(id, text)`` sorted by id -- the same
        contract the Decode workbench palette consumes for every backend.
        """
        out: list[tuple[int, str]] = []
        decoder = getattr(self.tokenizer, "added_tokens_decoder", None)
        if decoder:
            for tid, added in decoder.items():
                if getattr(added, "special", False):
                    out.append((int(tid), str(getattr(added, "content", added))))
        else:
            ids = list(getattr(self.tokenizer, "all_special_ids", []) or [])
            toks = list(getattr(self.tokenizer, "all_special_tokens", []) or [])
            for tid, txt in zip(ids, toks):
                out.append((int(tid), str(txt)))
        out.sort(key=lambda pair: pair[0])
        return out

    def _logprobs_at_last(self, token_ids: list[int]):
        torch = self._torch
        input_ids = torch.tensor([token_ids], device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0, -1]
        return torch.log_softmax(logits.float(), dim=-1)

    def next_distribution(
        self,
        token_ids: list[int],
        top_k: int,
        *,
        watch_ids: Sequence[int] = (),
    ) -> StepResult:
        torch = self._torch
        logp = self._logprobs_at_last(token_ids)
        k = min(top_k, logp.shape[-1])
        vals, idx = torch.topk(logp, k)
        cands = [
            TokenCandidate(
                int(i),
                self.piece(int(i)),
                float(v),
                rank,
                is_special=self._is_special(int(i)),
            )
            for rank, (v, i) in enumerate(zip(vals.tolist(), idx.tolist()))
        ]
        step = StepResult(position=len(token_ids), candidates=cands, is_full_vocab=True)
        # Full-vocab backend: read the EXACT logprob of each watched id
        # from the same forward-pass tensor, including ids that fell
        # outside the requested top_k (i.e. tail-of-distribution
        # probes -- the only way to see P(EOS) on a confident model
        # without raising top_k to vocab_size).
        for wid in watch_ids:
            wid_i = int(wid)
            step.watched[wid_i] = self._exact_candidate(logp, wid_i)
        return step

    def _exact_candidate(self, dist, token_id: int) -> TokenCandidate:
        lp = float(dist[token_id].item())
        rank = int((dist > dist[token_id]).sum().item())
        return TokenCandidate(
            token_id,
            self.piece(token_id),
            lp,
            rank,
            is_special=self._is_special(token_id),
        )

    def verify_greedy(
        self, context_ids: list[int], draft_ids: list[int]
    ) -> tuple[int, TokenCandidate]:
        """Verify drafted tokens against greedy target in ONE forward pass.

        Returns (accepted, correction): how many leading draft tokens match the
        target's greedy choice, and the target token that replaces the first
        mismatch -- or, if all drafts are accepted, the bonus next token. This is
        what gives speculative decoding its speedup: gamma tokens verified per
        single target forward pass.
        """
        torch = self._torch
        full = list(context_ids) + list(draft_ids)
        input_ids = torch.tensor([full], device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0]
        base = len(context_ids) - 1
        accepted = 0
        for i in range(len(draft_ids)):
            pos = base + i
            tgt = int(torch.argmax(logits[pos]).item())
            if tgt == draft_ids[i]:
                accepted += 1
            else:
                dist = torch.log_softmax(logits[pos].float(), dim=-1)
                return accepted, TokenCandidate(
                    tgt,
                    self.piece(tgt),
                    float(dist[tgt]),
                    0,
                    is_special=self._is_special(tgt),
                )
        # all accepted -> emit the bonus token from the last position
        pos = len(full) - 1
        tgt = int(torch.argmax(logits[pos]).item())
        dist = torch.log_softmax(logits[pos].float(), dim=-1)
        return accepted, TokenCandidate(
            tgt,
            self.piece(tgt),
            float(dist[tgt]),
            0,
            is_special=self._is_special(tgt),
        )

    def score_prompt(
        self,
        prompt: str,
        top_k: int,
        watch_ids: list[int] | None = None,
        *,
        prepend_token_ids: Sequence[int] = (),
    ) -> list[StepResult]:
        """Whole-context inspection, including the trailing "what comes next?".

        For an N-token prompt this returns N StepResults. The first N-1
        rows compare each logit row against the *actual* next token in the
        prompt (``chosen != None``). The final row -- the distribution
        conditioned on the whole prompt -- has ``chosen=None`` since there
        is no ground-truth next token; that's the row that answers "did
        the model want to stop here?". The same forward pass produces all
        N rows for free, so this is no slower than the old behaviour.

        ``prepend_token_ids`` lets callers seed the sequence with extra
        tokens BEFORE the tokenized prompt (typically the model's BOS
        marker) so the user can observe what the model would predict for
        position 0 of their prompt -- an otherwise-unscorable position
        because autoregressive models compute ``P(next | prior)`` and the
        first token has no prior. The prepended tokens become the leading
        StepResults; the row whose ``chosen`` is the user's first prompt
        token is the answer to "what does the model expect to see after
        my prepend".
        """
        torch = self._torch
        watch_ids = watch_ids or []
        prepend_ids = [int(t) for t in (prepend_token_ids or [])]
        prompt_ids = self.tokenize(prompt)
        ids = prepend_ids + list(prompt_ids)
        if not ids:
            return []
        input_ids = torch.tensor([ids], device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0]  # [seq, vocab]
        logp = torch.log_softmax(logits.float(), dim=-1)
        results: list[StepResult] = []
        for i in range(len(ids)):
            dist = logp[i]
            k = min(top_k, dist.shape[-1])
            vals, idx = torch.topk(dist, k)
            cands = [
                TokenCandidate(
                    int(j),
                    self.piece(int(j)),
                    float(v),
                    rank,
                    is_special=self._is_special(int(j)),
                )
                for rank, (v, j) in enumerate(zip(vals.tolist(), idx.tolist()))
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


def _gather_eos_ids(model_config, tokenizer) -> tuple[int, ...]:
    """Pull EOS ids from both the model config and the tokenizer.

    A modern HF model may advertise multiple EOS ids (Qwen, Llama-3 family
    all do this for chat templates). We coerce ``None`` / scalar / list /
    tuple shapes into a unique tuple of ints, in stable order. If the
    tokenizer's ``eos_token_id`` differs from the model's, we include both
    so the generation loop catches either.
    """
    out: list[int] = []
    seen: set[int] = set()

    def _absorb(v):
        if v is None:
            return
        if isinstance(v, (list, tuple)):
            for x in v:
                _absorb(x)
            return
        try:
            tid = int(v)
        except (TypeError, ValueError):
            return
        if tid not in seen:
            seen.add(tid)
            out.append(tid)

    _absorb(getattr(model_config, "eos_token_id", None))
    _absorb(getattr(tokenizer, "eos_token_id", None))
    return tuple(out)


def _gather_bos_ids(model_config, tokenizer) -> tuple[int, ...]:
    """Pull BOS ids from both the model config and the tokenizer.

    Mirror of ``_gather_eos_ids`` for the begin-of-sequence marker. Most
    HF models expose a single ``bos_token_id``; we collect both the
    config and tokenizer values in case they diverge. Returns an empty
    tuple when the model has no canonical BOS (e.g. some GPT-2 variants)
    -- the UI uses that to grey out the "fill BOS" helper.
    """
    out: list[int] = []
    seen: set[int] = set()

    def _absorb(v):
        if v is None:
            return
        if isinstance(v, (list, tuple)):
            for x in v:
                _absorb(x)
            return
        try:
            tid = int(v)
        except (TypeError, ValueError):
            return
        if tid not in seen:
            seen.add(tid)
            out.append(tid)

    _absorb(getattr(model_config, "bos_token_id", None))
    _absorb(getattr(tokenizer, "bos_token_id", None))
    return tuple(out)
