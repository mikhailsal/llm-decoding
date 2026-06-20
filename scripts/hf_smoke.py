"""Wave-0 HuggingFace white-box smoke test.

Loads a base model in 4-bit NF4 (device_map="auto" -> GPU+CPU) and demonstrates
the two things the transformers backend uniquely gives us:

1. Full next-token distribution (softmax over the WHOLE vocabulary).
2. Whole-context logits: one distribution per prompt position. We show the
   teacher-forced probability the model assigned to each actual next token,
   which is exactly what "inspect" mode will visualize later.

If the primary (Qwen3.5-9B-Base hybrid/multimodal) model fails to load or run on
the Pascal P40, we fall back to a small dense base model so the white-box path
is still usable.

Run on dsbx-host:
  source .venv/bin/activate && source scripts/env_wind.sh
  python scripts/hf_smoke.py
"""

from __future__ import annotations

import argparse
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PRIMARY = "Qwen/Qwen3.5-9B-Base"
FALLBACK = "Qwen/Qwen3-1.7B-Base"
PROMPT = "The capital of France is"


def load(model_id: str, four_bit: bool, gpu_mem: str = "4500MiB", cpu_mem: str = "13GiB"):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs = dict(device_map="auto", trust_remote_code=True)
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            # Double-quant + CPU offload triggers a bnb/accelerate meta-tensor
            # bug (offset.item() on meta) on this box, so keep it off.
            bnb_4bit_use_double_quant=False,
            # Required on the 6 GB P40: the 9B 4-bit weights don't fully fit in
            # VRAM, so accelerate spills overflow modules to CPU. Without this
            # flag bnb refuses any CPU/disk dispatch.
            llm_int8_enable_fp32_cpu_offload=True,
        )
        # Cap GPU so device_map="auto" knows to offload the remainder to CPU RAM
        # instead of OOMing. Leaves ~1.5 GB VRAM headroom for activations/KV.
        kwargs["max_memory"] = {0: gpu_mem, "cpu": cpu_mem}
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tok, model


def demo(model_id: str, four_bit: bool) -> bool:
    print(f"\n=== Loading {model_id} (4bit={four_bit}) ===", flush=True)
    t0 = time.time()
    tok, model = load(model_id, four_bit)
    print(f"loaded in {time.time() - t0:.1f}s; device_map sample:", flush=True)
    try:
        dm = getattr(model, "hf_device_map", {})
        print("  layers on:", sorted({str(v) for v in dm.values()}))
    except Exception:  # noqa: BLE001
        pass

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    ids = input_ids[0].tolist()

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits  # [1, seq, vocab]
    print(f"forward in {time.time() - t0:.2f}s; logits shape = {tuple(logits.shape)}")
    vocab = logits.shape[-1]

    # (1) Full next-token distribution from the last position.
    last = torch.log_softmax(logits[0, -1].float(), dim=-1)
    topv, topi = torch.topk(last, 8)
    print(f"\nFull-vocab next-token after {PROMPT!r} (vocab={vocab}):")
    for lp, idx in zip(topv.tolist(), topi.tolist()):
        print(f"  {tok.decode([idx])!r:>14}  p={torch.exp(torch.tensor(lp)).item():6.2%}  logprob={lp:7.3f}")

    # (2) Whole-context: prob the model gave to each ACTUAL next token.
    logp = torch.log_softmax(logits[0].float(), dim=-1)  # [seq, vocab]
    print("\nWhole-context (teacher forcing), p(next actual token | context):")
    for i in range(len(ids) - 1):
        nxt = ids[i + 1]
        p = torch.exp(logp[i, nxt]).item()
        print(f"  pos {i:>2} {tok.decode([ids[i]])!r:>10} -> {tok.decode([nxt])!r:>10}  p={p:6.2%}")

    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PRIMARY)
    ap.add_argument("--fallback", default=FALLBACK)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    try:
        demo(args.model, four_bit=not args.no_4bit)
        print("\nPRIMARY OK")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\nPRIMARY FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print(f"\n--- Falling back to dense base {args.fallback} ---")
        try:
            demo(args.fallback, four_bit=not args.no_4bit)
            print("\nFALLBACK OK")
            return 0
        except Exception as exc2:  # noqa: BLE001
            print(f"\nFALLBACK FAILED: {type(exc2).__name__}: {exc2}")
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
