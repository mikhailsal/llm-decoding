# LLM Decoding Sandbox

A console-first, library-backed tool for **studying how language models assign
probabilities to tokens** and how **decoding/sampling** turns those
probabilities into text. It is an educational/research project: inspect the
distribution at every position, plug in your own decoding functions, drive
decoding token-by-token by hand, and (later) experiment with speculative
decoding -- against both **local models** and **logprob-capable cloud
providers**.

## The mental model

At each position a model produces a score (logit) for every token in its
vocabulary; a softmax turns those into a probability distribution. "Decoding" is
just repeatedly choosing the next token from that distribution. What you can see
depends on the backend:

- **Whole-context logits (every prompt token):** complete only with local
  **HuggingFace transformers** (full vocabulary). Available top-k via
  **Fireworks** `echo`.
- **Full-vocabulary distribution + custom samplers + true manual stepping +
  speculative:** **transformers** (the "white box").
- **Cloud generated-token top-k:** Fireworks (<=5), NVIDIA NIM (<=20),
  OpenRouter (<=20).

### HuggingFace `transformers`, briefly

`transformers` runs the forward pass yourself: one call returns
`outputs.logits` of shape `[batch, seq_len, vocab_size]` -- the full
distribution at **every** position, prompt included. From that single tensor you
get whole-context inspection (softmax over the vocab axis), custom decoders
(operate directly on logits), manual decoding (keep `past_key_values`, append any
chosen token id, step again), and speculative decoding (assisted generation). It
is the only backend that exposes the entire vocabulary instead of a top-k slice.

## Provider logprob support (live-verified, June 2026)

| Provider | chat logprobs | whole-context (prompt) logprobs | notes |
|---|---|---|---|
| **Fireworks** | yes (top_logprobs <= 5) | **yes** (`/completions` `echo`) | frontier models (gpt-oss-120b, glm-5, kimi, deepseek) |
| **NVIDIA NIM** | yes (top_logprobs <= 20) | no (no `/completions`) | hosts Qwen3.5 MoE siblings |
| **OpenRouter** | yes (needs `provider.require_parameters`) | no | routes to a capable provider |
| **LM Studio** | yes (top_logprobs <= 10) | no (chat-only by default) | local OpenAI-compatible server, no key needed |
| **Gemini AI Studio** | no (capability gate) | no | use Vertex AI + billing if ever needed |
| Ollama 0.7.0 | no | no | not used |

## Where things run

- **All model compute runs on `dsbx-host`** (Ubuntu Linux on the Windows main PC,
  NVIDIA P40 6 GB). The **client is only the editor/control box.**
- Edit here, then `make sync` (rsync over SSH) and run on `dsbx-host`.

## Storage (important)

The Linux ext4 disk is a sparse `.img` on `C:` (local SSD, tight on space), so
"911 GB free" inside Linux is misleading. Strategy:

- Active model files (GGUF, 4-bit weights): ext4 inside Linux `/`.
- Bulk caches (`HF_HOME`, pip): `~/.cache/dsbx` (large local SSD).
- **Never use `R:`** (unreliable) or `S:` (HDD).
- `dsbx doctor` runs a free-space preflight before any heavy work.

## Quickstart

### On the client (editor + cloud probes)

```bash
cd ~/projects/llm-decoding
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp config.example.toml config.toml   # optional; defaults are fine
dsbx doctor          # checks keys + disk space
dsbx probe           # live provider logprob check
```

API keys are read from the environment. `config.toml` points
`secrets_env_file` at `~/.config/dsbx/secrets.env`, which holds
`FIREWORKS_API_KEY`, `NVIDIA_API_KEY`, `OPENROUTER_API_KEY`.

### On `dsbx-host` (local models)

```bash
make sync                       # push source from the client
ssh dsbx-host
cd llm-decoding
bash scripts/setup_wind.sh      # venv on ext4, caches on ~/.cache/dsbx, torch+transformers
source .venv/bin/activate
dsbx doctor                     # should now show torch cuda=True on the P40
```

## Commands

- `dsbx doctor` -- environment, API keys, disk free-space preflight, local engines.
- `dsbx probe` -- live provider logprob capability check.
- `dsbx inspect "<text>"` -- per-token confidence + watch-token highlighting
  (whole-context). `--backend hf|llamacpp|fireworks|nim|openrouter|lmstudio`,
  `--watch ' Paris'`, `--top-k`, `--candidates N`.
- `dsbx generate "<text>"` -- decode with a sampler, showing per-step changes vs
  greedy. `--sampler greedy|temperature|top_k|top_p|min_p|typical|custom`,
  `--temperature`, `--top-p`, `--min-p`, `--typical-p`, `--custom-file f.py:fn`,
  `--stop ' END'` (repeatable single-token early-stop).
- `dsbx manual "<text>"` -- interactive token-by-token TUI (pick by rank, force
  any token, undo, save/load transcript).
- `dsbx spec "<text>"` -- speculative decoding (HF draft+target) with
  accept/reject visualization and a tokens-per-target-pass speedup metric.

### Examples

```bash
# Whole-context inspection on the local 9B base via llama.cpp
dsbx inspect "The capital of France is Paris" --backend llamacpp --watch " Paris"

# Full-vocab inspection (exact prob/rank for ANY token) via HF
dsbx inspect "The capital of France is Paris" --backend hf --watch " London"

# Whole-context inspection on a frontier cloud model (Fireworks echo)
dsbx inspect "The capital of France is Paris" --backend fireworks

# Compare samplers
dsbx generate "Once upon a time" --backend llamacpp --sampler top_p --top-p 0.9 --temperature 1.1

# Your own decoding function
dsbx generate "Once upon a time" --backend hf \
  --sampler custom --custom-file examples/custom_sampler.py:decode

# Speculative decoding (HF)
dsbx spec "The capital of France is" --gamma 4
```

## Layout

```
decoding_sandbox/
  core/        config, storage, provider_probe, types, backend, factory,
               samplers, engine, manual, speculative  (no UI deps)
  backends/    hf (full vocab) / llamacpp (top-k) / openai_compat (cloud)
  cli/         argparse front-end, rich rendering, manual TUI
scripts/       sync_to_wind.sh, setup_wind.sh, build_llamacpp_wind.sh,
               run_llama_server_wind.sh, hf_smoke.py, test_manual.py
examples/      custom_sampler.py
```

## What was verified on the hardware (Wave 0)

- **llama.cpp + CUDA** built for the P40 (sm_61, g++-12 host compiler). The
  Qwen3.5-9B-Base Q4_K_M GGUF loads (hybrid Gated DeltaNet), `-ngl 20`/ctx 4096
  uses ~3.6 GB VRAM at ~11 tok/s, and `/completion n_probs` returns top-k logprobs.
- **HF transformers** white-box engine works with a dense base (full vocab
  151936, whole-context teacher forcing). The 9B base does **not** load in 4-bit
  on the 6 GB Pascal (bitsandbytes+accelerate meta-tensor bug on the hybrid arch),
  so the 9B base is served by llama.cpp and HF uses a dense base.
- **Cloud**: Fireworks chat top_logprobs + whole-context echo (frontier models);
  NIM and OpenRouter generated-token logprobs (OpenRouter needs
  `require_parameters`); Gemini AI Studio deferred (logprobs gated off).

## Status

All planned waves (0-5) are implemented. Foundations/environment, backend
abstraction + `inspect`, samplers + `generate` (with `--stop`), the manual TUI,
cloud backends (Fireworks `echo` whole-context + chat-only NIM/OpenRouter/LM
Studio), and speculative decoding via the `Speculator` Protocol +
`HFSpeculator` (HF assisted-generation style; a `LlamaCppSpeculator` matching
the same Protocol is the natural next addition). All heavy commands run the
`storage.preflight_or_raise` disk check first (bypass with `--skip-preflight`).
Next up: a thin FastAPI + browser UI over the same `core/`.
