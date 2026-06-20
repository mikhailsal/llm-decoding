"""HuggingFace transformers backend: the full-vocabulary white box.

One forward pass yields logits of shape ``[1, seq, vocab]`` -- the exact
distribution at every position. This is the only backend that returns true
full-vocab probabilities, exact ranks, and whole-context inspection in a single
pass, and it can force arbitrary tokens (for manual/speculative decoding).

torch/transformers are imported lazily so the CLI still runs on machines without
them (e.g. the client); this backend is only instantiated on dsbx-host.
"""

from __future__ import annotations

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
            print(f"[HFBackend] {model_id} failed ({type(exc).__name__}); "
                  f"falling back to {fallback_model}")
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
        )

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, return_tensors=None)["input_ids"]

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def piece(self, token_id: int) -> str:
        if token_id not in self._piece_cache:
            self._piece_cache[token_id] = self.tokenizer.decode([token_id])
        return self._piece_cache[token_id]

    def _logprobs_at_last(self, token_ids: list[int]):
        torch = self._torch
        input_ids = torch.tensor([token_ids], device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0, -1]
        return torch.log_softmax(logits.float(), dim=-1)

    def next_distribution(self, token_ids: list[int], top_k: int) -> StepResult:
        torch = self._torch
        logp = self._logprobs_at_last(token_ids)
        k = min(top_k, logp.shape[-1])
        vals, idx = torch.topk(logp, k)
        cands = [
            TokenCandidate(int(i), self.piece(int(i)), float(v), rank)
            for rank, (v, i) in enumerate(zip(vals.tolist(), idx.tolist()))
        ]
        return StepResult(position=len(token_ids), candidates=cands, is_full_vocab=True)

    def _exact_candidate(self, dist, token_id: int) -> TokenCandidate:
        lp = float(dist[token_id].item())
        rank = int((dist > dist[token_id]).sum().item())
        return TokenCandidate(token_id, self.piece(token_id), lp, rank)

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
                return accepted, TokenCandidate(tgt, self.piece(tgt), float(dist[tgt]), 0)
        # all accepted -> emit the bonus token from the last position
        pos = len(full) - 1
        tgt = int(torch.argmax(logits[pos]).item())
        dist = torch.log_softmax(logits[pos].float(), dim=-1)
        return accepted, TokenCandidate(tgt, self.piece(tgt), float(dist[tgt]), 0)

    def score_prompt(
        self, prompt: str, top_k: int, watch_ids: list[int] | None = None
    ) -> list[StepResult]:
        torch = self._torch
        watch_ids = watch_ids or []
        ids = self.tokenize(prompt)
        input_ids = torch.tensor([ids], device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0]  # [seq, vocab]
        logp = torch.log_softmax(logits.float(), dim=-1)
        results: list[StepResult] = []
        for i in range(len(ids) - 1):
            dist = logp[i]
            k = min(top_k, dist.shape[-1])
            vals, idx = torch.topk(dist, k)
            cands = [
                TokenCandidate(int(j), self.piece(int(j)), float(v), rank)
                for rank, (v, j) in enumerate(zip(vals.tolist(), idx.tolist()))
            ]
            chosen = self._exact_candidate(dist, ids[i + 1])
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
