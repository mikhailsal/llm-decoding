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

- **Whole-context logits (every prompt token):** complete with local
  **HuggingFace transformers** *and* the in-process
  **`llamacpp-py`** backend (full vocabulary). Available top-k via
  **Fireworks** `echo`.
- **Full-vocabulary distribution + custom samplers + true manual stepping +
  speculative:** **transformers** *or* **`llamacpp-py`** (both expose the
  whole `[seq, vocab]` logits tensor). HF runs PyTorch, `llamacpp-py` runs the
  llama.cpp engine in-process on the same Q4 GGUF.
- **Top-k only (no full vocab):** **`llamacpp`** HTTP backend (top-k = N
  candidates per position) and cloud chat-only providers.
- **Cloud generated-token top-k:** Fireworks (<=5), NVIDIA NIM (<=20),
  OpenRouter (<=20), LM Studio (<=10).

### HuggingFace `transformers`, briefly

`transformers` runs the forward pass yourself: one call returns
`outputs.logits` of shape `[batch, seq_len, vocab_size]` -- the full
distribution at **every** position, prompt included. From that single tensor you
get whole-context inspection (softmax over the vocab axis), custom decoders
(operate directly on logits), manual decoding (keep `past_key_values`, append any
chosen token id, step again), and speculative decoding (assisted generation).

### `llamacpp-py`, briefly

The in-process `llama-cpp-python` binding compiled against the same
sm_61 CUDA build as `llama-server`. Initialized with `logits_all=True`, it
exposes the equivalent of HF's `outputs.logits` -- a `[seq, vocab]`
matrix via `Llama.scores` -- for **GGUF models that HF can't load** (e.g.
Qwen3.5-9B-Base on the 6 GB P40 due to the bitsandbytes + accelerate
meta-tensor bug on its hybrid arch). Same Q4 GGUF on disk as `llamacpp`
(HTTP) -- no extra download -- and the same partial GPU offload via
`n_gpu_layers`. KV-cache is reused when subsequent calls extend the
previous context (manual stepping stays cheap).

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
  (whole-context).
  `--backend hf|llamacpp|llamacpp-py|fireworks|nim|openrouter|lmstudio`,
  `--watch ' Paris'`, `--top-k`, `--candidates N`.
- `dsbx generate "<text>"` -- decode with a sampler, showing per-step changes vs
  greedy. `--sampler greedy|temperature|top_k|top_p|min_p|typical|custom`,
  `--temperature`, `--top-p`, `--min-p`, `--typical-p`, `--custom-file f.py:fn`,
  `--stop ' END'` (repeatable single-token early-stop).
- `dsbx manual "<text>"` -- interactive token-by-token TUI (pick by rank, force
  any token, undo, save/load transcript).
- `dsbx spec "<text>"` -- speculative decoding (HF draft+target) with
  accept/reject visualization and a tokens-per-target-pass speedup metric.
- `dsbx session` -- long-lived REPL that **keeps the model loaded across
  commands**. The 30+s GGUF load for the 9B happens once; every subsequent
  `inspect`/`generate`/`manual` in the session runs immediately. Meta
  commands: `:caps`, `:backend NAME [MODEL]` (swaps the loaded model),
  `:timing on|off`, `:history`, `:help`, `:quit`.

Every heavy command prints a one-line timing summary
(`timing: prompt eval ... | decode ... | total ...`), with tokens-per-second
for any phase where the divisor is meaningful. Suppress with `--no-timing`
(or `:timing off` in a session).

### Colors over SSH

By default `dsbx` uses the standard `rich` TTY detection: ANSI escape codes
are emitted when stdout is a terminal and stripped otherwise. That's the
right choice when piping to a file, but it strips every confidence-level
color and special-token highlight when you run

```bash
ssh dsbx-host 'dsbx inspect ...'
```

(non-interactive SSH -- stdout isn't a TTY on the remote side). Use one
of these to keep colors:

- `dsbx --color always inspect ...` -- explicit per-invocation override.
- `FORCE_COLOR=1 dsbx inspect ...` -- standard env var, respected by
  rich.
- `ssh -t dsbx-host 'dsbx inspect ...'` -- request a remote PTY; the
  terminal then looks like a TTY to rich and `--color auto` works.

`--color never` (or `NO_COLOR=1`) disables colors even on a TTY.

### Token rendering

Tokens that look identical in a column actually differ by leading/trailing
whitespace -- ``"I"``, ``" I"`` and ``"I "`` are three different ids. The
renderer surfaces this with explicit markers:

- Leading / trailing spaces -> `␣` (one per space), so ` I` reads as `␣I`
  and `I ` reads as `I␣`. Internal spaces in prose stay untouched.
- Newline -> `↵`, tab -> `→`, other control bytes -> `\xNN`.
- Empty token -> `<empty>` (dim).
- Special tokens (EOS/BOS/PAD/`<|im_start|>`/`<|endoftext|>`, anything the
  tokenizer marks special or any text matching `<|...|>`) render in magenta
  bold.

### EOS and stopping

Each backend reports its end-of-text token ids in
`Capabilities.eos_token_ids`:

- **HF transformers**: read from `model.config.eos_token_id` and
  `tokenizer.eos_token_id` (both, deduped -- modern models like Qwen list
  several for chat templates).
- **`llamacpp-py`**: read from `Llama.token_eos()` plus `Llama.token_eot()`
  when the binding exposes it (Qwen-style chat templates).
- **`llamacpp` HTTP** and **OpenAI-compat providers**: the server API
  doesn't expose this, so EOS detection is unavailable.

`dsbx generate` honors EOS by default: the moment the sampler picks an
EOS id, the loop stops and the footer reads
`stopped on EOS: model emitted <|endoftext|> (id=...)`. The footer also
reports `stopped on --stop token: ...` and
`reached --max-tokens=... (model did not emit EOS).` so it's always clear
*why* generation halted. Set `respect_eos=False` at the engine level (or
extend the CLI later) to probe what the model would emit past EOS.

`dsbx inspect` (and `:caps` inside a session) prints the configured EOS
ids in the banner, e.g.
`EOS ids: 248044=<special>` -- the magenta marker means the token's
printable form is empty (a true control token), so when *that* id shows
up at a generation step the model is actually emitting EOS rather than a
visible string. Typing `<|endoftext|>` literally in your prompt does not
get tokenized as the EOS id -- the binding's `tokenize()` BPE-encodes the
literal characters; the real EOS id only appears when the model itself
chooses it.

### Recipe: track P(EOS) across a fixed context

Because EOS often detokenizes to empty/unprintable text, `--watch ' Paris'`
can't reach it -- there's no string to type. Use one of:

```bash
# Pull every EOS id from capabilities and add a column per id.
dsbx inspect --watch-eos 'The weather today is surprisingly dry.'

# Or pin a specific id (any reserved/control token, not just EOS):
dsbx inspect --watch-id 248044 --watch-id 151643 'A short test.'

# Mix freely with text watches:
dsbx inspect --watch ' Paris' --watch-eos --watch-id 1234 'France's capital is'
```

What you get:

- One extra column per watched id, header reads e.g. `watch EOS:248044`
  or `watch id=1234 ' Paris'` (the piece is appended for sanity).
- Each row shows the **exact** logprob+rank for that id at that position
  -- even on full-vocab backends, even if the id falls outside the
  top-k for that step (`watch_ids` queries the distribution directly).
- The inspection table includes a trailing **predict-next** row,
  visibly marked ``N (next)`` in the position column, that shows what
  the model would emit *after* the entire prompt. For the example
  prompt ending in a period, this is precisely where you see how
  strongly the model "wants to finish": the EOS column on the ``(next)``
  row is the answer.
- Duplicates across `--watch` / `--watch-id` / `--watch-eos` are deduped
  by id, so the same column doesn't appear twice.
- Backends without EOS info (HTTP llama.cpp, cloud providers) print a
  yellow warning and the `--watch-eos` column is simply omitted.

Inside a `dsbx session` the same flags work after the `inspect` keyword:
`inspect --watch-eos "<prompt>"`, with the model staying loaded between
queries so you can scan many prompts cheaply.

### Examples

```bash
# Whole-context inspection on the local 9B base via llama.cpp (top-k)
dsbx inspect "The capital of France is Paris" --backend llamacpp --watch " Paris"

# FULL VOCAB inspection of the 9B GGUF in-process (white-box; same Q4 weights)
dsbx inspect "The capital of France is Paris" --backend llamacpp-py --watch " London"

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

# Persistent REPL: pay the 9B GGUF load once, then iterate cheaply
dsbx session --backend llamacpp-py
# inside the session prompt (`dsbx> `):
#   inspect "The weather today is surprisingly dry." --watch ' dry'
#   generate "Once upon a time" --sampler top_p --top-p 0.9 --max-tokens 30
#   :backend hf   # swap to HF transformers; closes the 9B, loads the dense base
#   :caps         # show the new backend's capabilities
#   :quit
```

## Layout

```
decoding_sandbox/
  core/        config, storage, provider_probe, types, backend, factory,
               samplers, engine, manual, speculative  (no UI deps)
  backends/    hf (full vocab, PyTorch) /
               llamacpp (top-k via HTTP) /
               llamacpp-py (full vocab via in-process llama-cpp-python) /
               openai_compat (cloud)
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
  so the 9B base is served by `llamacpp`/`llamacpp-py` and HF uses a dense base.
- **`llamacpp-py`** (in-process `llama-cpp-python` with `logits_all=True`)
  closes the gap: the 9B Q4 GGUF runs on the same sm_61 build and exposes the
  FULL `[seq, vocab]` logits tensor, so `inspect`/`generate`/`manual`/`spec`
  get the same white-box features HF gives for smaller models.
- **Cloud**: Fireworks chat top_logprobs + whole-context echo (frontier models);
  NIM and OpenRouter generated-token logprobs (OpenRouter needs
  `require_parameters`); Gemini AI Studio deferred (logprobs gated off).

## Status

All planned waves (0-5) are implemented. Foundations/environment, backend
abstraction + `inspect`, samplers + `generate` (with `--stop` *and* native
EOS handling), the manual TUI, cloud backends (Fireworks `echo` whole-context
+ chat-only NIM/OpenRouter/LM Studio), and speculative decoding via the
`Speculator` Protocol + `HFSpeculator`. Post-plan additions:

- **`llamacpp-py`** in-process backend (full vocab via `Llama.scores` with
  `logits_all=True`), which gives the 9B Qwen3.5 base the same white-box
  experience as HF does for smaller models.
- **`dsbx session`** REPL that keeps the loaded backend alive across
  multiple commands; one-time 30 s model load is amortized.
- **Per-command timing/TPS summary** on every heavy path.
- **Visible-whitespace token rendering** (`␣` for leading/trailing spaces,
  `↵` newline, `→` tab, `<empty>` / `<special>`) and magenta highlighting
  for special tokens, so ``"I"`` / ``" I"`` / ``"I "`` never collapse in a
  column.
- **EOS surfaced in capabilities** (`Capabilities.eos_token_ids`,
  populated by HF and `llamacpp-py`); `generate` stops when the model
  emits an EOS id and the footer prints `stopped on EOS: ... (id=...)`.

All heavy commands run the `storage.preflight_or_raise` disk check first
(bypass with `--skip-preflight`). Next up: a thin FastAPI + browser UI over
the same `core/`, and a `LlamaCppSpeculator` mirroring the existing
`HFSpeculator`.
