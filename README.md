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
| **NVIDIA NIM** | yes (top_logprobs <= 20) | no (no `/completions`) | **registered, generation disabled** -- chat-only; see Decode Workbench Phase 0 |
| **OpenRouter** | yes (needs `provider.require_parameters`) | no | **registered, generation disabled** -- chat-only; see Decode Workbench Phase 0 |
| **LM Studio** | yes (top_logprobs <= 10) | no (chat-only by default) | local OpenAI-compatible server, no key needed |
| **Gemini AI Studio** | no (capability gate) | no | use Vertex AI + billing if ever needed |
| Ollama 0.7.0 | no | no | not used |

> **Chat-only providers are gated off** (NIM, OpenRouter). Their `next_distribution`
> used to silently route through `/chat/completions` with a growing `[{role: user,
> content: detokenize(prompt + emitted_so_far)}]` message on every step, so the
> "continuation" we displayed was actually N independent first-responses to N
> slightly-different user queries -- not a real continuation, and not what an
> educational sandbox is supposed to show. The backends remain registered (so
> `/api/v1/info` still lists them and the frontend backend picker shows them as
> a disabled option with a tooltip), but the decode workbench refuses to start a
> generate / inspect / manual session against them with a 400 carrying the same
> explanation. Re-enable once a proper chat-mode UI (system / user / assistant
> turns, real `/chat/completions` wire shape, no per-step inspection inside an
> assistant turn) lands -- tracked as a separate PR.

## Where things run

- **All model compute runs on `dsbx-host`** (Ubuntu Linux on the Windows main PC,
  NVIDIA P40 6 GB).
- The **client runs the TUI** (and any cloud-backend traffic, since it
  has the VPN). It connects to `dsbx-host` over HTTP via the long-lived
  `dsbx serve` process -- so the 9B GGUF / HF model load happens once on
  `dsbx-host` and stays warm across every command. See
  [Running the server on `dsbx-host`](#running-the-server-on-dsbx-host) below.
- Edit here, then `make sync` (rsync over SSH) and (re)start the server
  on `dsbx-host` only when its source changed.

## Storage (important)

The Linux ext4 disk is a sparse `.img` on `C:` (local SSD, tight on space), so
"911 GB free" inside Linux is misleading. Strategy:

- Active model files (GGUF, 4-bit weights): ext4 inside Linux `/`.
- Bulk caches (`HF_HOME`, pip): `~/.cache/dsbx` (large local SSD).
- **Never use `R:`** (unreliable) or `S:` (HDD).
- `dsbx doctor` runs a free-space preflight before any heavy work.

## Quickstart

### On the client (the TUI lives here)

```bash
cd ~/projects/llm-decoding
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                     # lightweight: rich, httpx, openai, ...
cp config.example.toml config.toml   # add a [remote.dsbx-host-py] block (see below)
dsbx doctor                          # checks keys + remote servers + disk
dsbx probe                           # live provider logprob check
dsbx inspect "The capital of France is" --backend dsbx-host-py
```

API keys are read from the environment. `config.toml` points
`secrets_env_file` at `~/.config/dsbx/secrets.env`, which holds
`FIREWORKS_API_KEY`, `NVIDIA_API_KEY`, `OPENROUTER_API_KEY`. Cloud
backends (Fireworks/NIM/OpenRouter/LM Studio) talk directly to the
provider from the client -- they don't go through the server.

### On `dsbx-host` (one-time setup for the model host)

```bash
make sync                       # push source from the client
ssh dsbx-host
cd llm-decoding
bash scripts/setup_wind.sh      # venv on ext4, caches on ~/.cache/dsbx, torch+transformers
source .venv/bin/activate
pip install -e ".[server]"      # adds fastapi + uvicorn
dsbx doctor                     # should now show torch cuda=True on the P40
```

### Running the server on `dsbx-host`

The server is a long-lived FastAPI process that loads one heavy backend
(`hf` or `llamacpp-py`) at startup and keeps it warm. The client's TUI
connects via HTTP/SSE, so the 30+ second GGUF load happens *once* per
server lifetime instead of once per `dsbx` invocation.

```bash
ssh dsbx-host
cd llm-decoding && source .venv/bin/activate

# Host the 9B Qwen3.5 Base GGUF (full-vocab via logits_all=True):
dsbx serve --backend llamacpp-py --host 0.0.0.0 --port 8000

# In a second shell on dsbx-host, host HF transformers in parallel:
dsbx serve --backend hf --host 0.0.0.0 --port 8001
```

`--host 0.0.0.0` is opt-in (a warning is printed) -- the default
`127.0.0.1` is loopback-only. The server has no auth, so keep this box
on a trusted LAN. A convenience launcher with the same defaults lives at
[scripts/run_dsbx_server_wind.sh](scripts/run_dsbx_server_wind.sh).

#### Swappable model slot (load / reload without a restart)

Each `dsbx serve` process now owns a *swappable model slot* rather than a
fixed model. The slot has a small state machine -- `empty` -> `loading`
-> `ready`, with any load failure landing in `error` (carrying the
message) -- and the loaded model can be changed at runtime:

```bash
# Start with no model loaded; pick one from the browser later:
dsbx serve --backend llamacpp-py --no-preload --host 0.0.0.0 --port 8000

# Or preload as before (default) and still allow later swaps.
dsbx serve --backend llamacpp-py --host 0.0.0.0 --port 8000
```

The server exposes three new endpoints used by the web UI (and scriptable
directly):

- `GET /v1/status` -- live slot state (`empty`/`loading`/`ready`/`error`),
  the loaded model, any error, and the capability envelope when ready.
- `GET /v1/models` -- the host's catalogue of *compatible* models: every
  `*.gguf` found under `[local.llamacpp_py].model_search_dirs` for the
  `llamacpp-py` kind, or the configured `[local.hf].models` list (unioned
  with `model` + `fallback_model`) for `hf`.
- `POST /v1/reload {"model": "<id>"}` -- close the current model and load
  `<id>` on a background thread (a 9B GGUF takes ~30 s). Returns
  immediately with `state: loading`; poll `/v1/status` until terminal.

Because the 6 GB P40 can't hold two 9B models at once, a reload closes
the old model *before* building the new one; a failed reload therefore
leaves the slot `empty`/`error` rather than falling back to the previous
model. Inference requests during `loading`/`empty`/`error` return HTTP
`409` with an explanatory `detail`.

### Connecting from the client

Add one `[remote.NAME]` block per server in `config.toml`. The name you
pick is what you'll pass to `--backend` (and what `[run].backend` can
default to):

```toml
[run]
backend = "dsbx-host-py"   # default backend when --backend is omitted

[remote.dsbx-host-py]
base_url = "http://192.0.2.42:8000"
# timeout = 120.0   # bump if the first request after model load is slow

[remote.dsbx-host-hf]
base_url = "http://192.0.2.42:8001"
```

Then on the client:

```bash
dsbx doctor                     # the 'Remote dsbx servers' table probes both
dsbx inspect "Hello there"      # uses run.backend = dsbx-host-py
dsbx generate "Once upon a time" --backend dsbx-host-py --sampler top_p --top-p 0.9
dsbx manual "The capital of"    --backend dsbx-host-hf
```

For `generate` the CLI auto-detects the server's streaming endpoint and
shows `(remote-stream)` in the sampler line; per-token rendering happens
as each SSE event arrives. Custom samplers (`--sampler custom`) keep
working but fall back to the per-step `next_distribution` loop, because
the server has no way to ingest arbitrary client code.

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
- `dsbx serve` -- run the HTTP server on `dsbx-host` that hosts one heavy
  backend (`--backend hf|llamacpp-py`) in a swappable model slot. The
  client's TUI connects via `[remote.NAME]` aliases. Add `--no-preload`
  to start with an empty slot and load a model on demand from the web UI.
  Requires `pip install -e ".[server]"`. See
  [Running the server on `dsbx-host`](#running-the-server-on-dsbx-host).
- `dsbx session` -- convenience REPL with command history and a single
  loaded backend. Meta commands: `:caps`, `:backend NAME [MODEL]`
  (swaps the loaded model), `:timing on|off`, `:history`, `:help`,
  `:quit`. (Historical note: this used to be the only way to amortize
  the 30+ s GGUF load across many commands. With `dsbx serve` in place
  it's now purely an ergonomic shell.)

Every heavy command prints a one-line timing summary
(`timing: prompt eval ... | decode ... | total ...`), with tokens-per-second
for any phase where the divisor is meaningful. Suppress with `--no-timing`
(or `:timing off` in a session).

### Colors over SSH (legacy footnote)

The TUI now runs **locally** on the client and talks to `dsbx-host` over
HTTP, so `rich`'s TTY detection works correctly and color rendering is
no problem. The `--color always|never` / `FORCE_COLOR` / `ssh -t`
workarounds are still in the binary (and documented in `dsbx --help`)
for the rare case where someone insists on `ssh dsbx-host 'dsbx inspect ...'`
-- but that path is no longer the recommended workflow.

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

On the OpenAI-compat path `respect_eos=False` used to be a silent lie
(no documented field to disable EOS halting on the server). That's now
fixed for providers that opt into the Fireworks-style `ignore_eos`
field: when `ProviderConfig.supports_ignore_eos` is true we ship
`ignore_eos: true` on the wire and the model actually keeps emitting
past its EOS. The browser's `respect EOS` checkbox is enabled on those
backends. Providers without the flag (NIM, OpenRouter, LM Studio
chat-only) still get the advisory note on the `usage` SSE frame so the
UI can say "the cloud ignored this flag" instead of pretending.

### Fireworks /v1/completions extensions

Fireworks exposes a small zoo of extension fields on `/v1/completions`
that turn the sandbox into a much better learning + debugging tool.
They are wired through `ProviderConfig.supports_*` flags so other
providers stay on the conservative OpenAI-compatible subset:

- `ignore_eos` -- see above. Unlocks the `respect EOS` checkbox.
- `perf_metrics_in_response` -- always-on. The provider returns a
  `perf_metrics` block (TTFT, prefill/generation durations, cached
  prompt tokens, backend host) that the middleware surfaces as a
  dedicated `perf` SSE frame; the browser renders it as a
  *server timings* panel right next to the running completion.
- `service_tier` -- per-request selector (`default` / `priority`). The
  UI shows the dropdown only when `Capabilities.supports_service_tier`
  is true.
- `prompt_cache_key` -- when set, requests sharing the same key are
  routed to the same backend replica to maximize KV-cache hit rate
  (great for manual decoding where the prefix is stable).
- `x-session-affinity` + `x-multi-turn-session-id` (HTTP headers) --
  when `session_id` is set on a request these two headers go on the
  wire; the second one enables Fireworks' **MoE Router Replay (R3)**
  feature, making MoE expert routing deterministic across turns of a
  multi-step session.

Phases 1-2 cover wire mechanics + expanded sampling
(`typical_p`, `mirostat`, repetition/frequency/presence penalties).
Phase 3 swapped the cloud-parser onto the **NewLogProbs** format
(`logprobs: true` + `top_logprobs: N` instead of a single integer):
- Token candidates now carry the **real model `token_id`** straight
  from the response, instead of the synthetic interned ids the legacy
  parser had to invent. That makes `--watch-id N` finally produce
  meaningful "what's the probability of token 1234?" traces against
  cloud providers.
- When `Capabilities.supports_sampling_mask` is true, the request also
  sets `sampling_mask: 'count'` and the response carries
  `sampling_mask_count` per position -- the number of tokens that
  survived the server's sampling filter stack. Both `/generate` and
  `/inspect` render this as a new **eligible** column right after the
  probability bar.

Phase 4 wired the remaining Fireworks-only knobs into both the request
and the UI:
- **`raw_output: true`** is always on for Fireworks. The provider's
  diagnostics block (`prompt_fragments`, `prompt_token_ids`,
  `grammar`, plus whatever else the upstream chose to emit) flows
  through a dedicated `raw_output` SSE frame and renders as a
  "what the model saw" panel beside the perf timings. Most useful
  when a custom chat template silently ate your system prompt --
  you see it directly instead of guessing from "why didn't the model
  follow the instructions?".
- **`logit_bias`** is a new field on `GenerateRequest`. The generate
  page exposes a row editor (token_id + bias in [-100, 100]) gated
  on `Capabilities.supports_logit_bias`. The wire shape matches
  OpenAI (`{"<token_id>": float}`); invalid rows are dropped silently
  on the client and again on the backend, so a single typo never
  fails the whole request. Use cases: ban a token (`-100`), nudge a
  rare option past a tight `top_p` (`+5..+15`), force a particular
  token in a grammar-constrained setup (`+100`).

Phase 5 collapses the "include prompt logits" workflow from two
network round trips into one when the backend advertises
`supports_combined_echo_stream`:
- A single `echo=true` + `stream=true` request returns BOTH the
  echoed per-prompt-token logprobs AND the streamed generated tokens
  in one connection. The frontend wire shape is unchanged
  (`prompt_score? -> step* -> perf? -> raw_output? -> usage -> done`)
  so existing consumers keep working without any client changes.
- A new `echo_last` field on `GenerateRequest` (Fireworks-specific,
  gated on `supports_echo_last`) restricts the echoed positions to
  the last N prompt tokens -- handy for long prompts where you only
  care about the trailing context. The generate page surfaces it as
  a small "echo last N" knob right under the *include prompt* check-
  box, visible only when the combined path is actually in use.
- `scripts/smoke_fireworks_echo_stream.py` is a one-shot diagnostic
  that POSTs an `echo + stream + max_tokens>0 + logprobs` body
  against the real Fireworks API and prints the chunk-by-chunk
  ordering, so you can confirm a new deployment tolerates the combo
  before flipping `supports_combined_echo_stream` on in
  `config.toml`.

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
               openai_compat (cloud) /
               remote (HTTP+SSE client for `dsbx serve`)
  server/      FastAPI app + pydantic wire schemas; `dsbx serve` entry
               (loads ONE backend per process and serves the Backend
               protocol over /v1/* + SSE for generate)
  cli/         argparse front-end, rich rendering, manual TUI, session REPL
scripts/       sync_to_wind.sh, setup_wind.sh, build_llamacpp_wind.sh,
               run_llama_server_wind.sh, run_dsbx_server_wind.sh,
               hf_smoke.py, test_manual.py
examples/      custom_sampler.py
```

### Client/server split (post-server architecture)

```
+-------------------+              +----------------------+
|  thinkpad (TUI)   |   HTTP+SSE   |  dsbx-host (dsbx serve)   |
|                   | <----------> |  - FastAPI + uvicorn |
|  - cli (TUI)      |              |  - one heavy backend |
|  - cloud backends |              |    (hf or llamacpp-py)
|  - RemoteBackend  |              |  - keeps model warm  |
+-------------------+              +----------------------+
        | HTTPS via VPN
        v
[ Fireworks / NIM / OpenRouter / LM Studio ]
```

`RemoteBackend` implements the same `Backend` protocol as the in-process
backends; every CLI command (`inspect`, `generate`, `manual`, `session`,
`spec`) works with it without branching. `generate` additionally uses
`stream_generate` for incremental SSE rendering when the backend
supports it.

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
- **`dsbx serve` HTTP server** (`decoding_sandbox/server/`) hosting one
  heavy backend per process, with a matching `RemoteBackend` client
  (`decoding_sandbox/backends/remote.py`). REST for tokenize / score /
  verify; SSE for `generate` (server runs the loop, client renders each
  step as the event arrives). Configured via `[remote.NAME]` blocks;
  `dsbx doctor` probes every configured server's `/v1/info`. This
  retires the "session = load-time hack" workaround and lays the wire
  protocol the next plan's browser UI will consume.

All heavy commands run the `storage.preflight_or_raise` disk check first
(bypass with `--skip-preflight`). Next up: a `LlamaCppSpeculator`
mirroring the existing `HFSpeculator`.

## Web UI

The browser UI is a SvelteKit single-page app served by `dsbx web` -- a
small FastAPI **middleware** that fronts every configured backend behind
a single bearer-token API. The browser never sees provider API keys, the
dsbx-host LAN address, or anything in `secrets_env_file`; it only knows the
address of `dsbx web` (and the token to talk to it).

```
+------------------+   bearer token       +--------------------+
| browser (SPA)    | ------------------>  | dsbx web           |
| svelte + ts      |  HTTP + SSE (auth)   | FastAPI on client|
|                  |                      |  - hides all keys  |
+------------------+                      |  - hides remote IPs|
                                          +--------------------+
                                              |          |
                       LAN HTTP+SSE           |          |  HTTPS keys
                       to dsbx-host/dsbx serve     v          v
                                  +----------------+  +------------------+
                                  | dsbx-host dsbx serve|  | Fireworks / NIM  |
                                  | (model loaded) |  | OpenRouter / LMS |
                                  +----------------+  +------------------+
```

The UI mirrors every TUI verb: `/inspect`, `/generate`, `/manual`,
`/spec`, `/status` (combines `doctor` + `probe`). Streaming token output
uses SSE end-to-end (`generate` and `spec`); manual-decoding state lives
on the middleware in a UUID-keyed, TTL-evicted session registry.

### Remote model control (Status page)

The `/status` page hosts a **Remote model control** card for every
`[remote.NAME]` backend. Each card shows the host's live slot state
(`no model loaded` / `loading…` / `ready` / `error`), a picker populated
from that host's compatible-model catalogue (the GGUFs on disk for a
`llamacpp-py` host, or the configured HF ids for an `hf` host), and a
Load/Reload button. Loading a 9B GGUF takes ~30 s; the card polls the
host every ~2 s while `loading` and refreshes the rest of the UI's
capability envelopes once the new model is `ready`. This pairs naturally
with `dsbx serve --no-preload`: start the host empty, then pick the model
from the browser.

The middleware proxies this through two scrubbed endpoints --
`GET /api/v1/backends/{name}/status` and
`POST /api/v1/backends/{name}/reload` -- which forward to the host's
`/v1/status` and `/v1/reload`. Errors are sanitized so the dsbx-host LAN
address never reaches the browser, and both endpoints 400 for non-remote
families (cloud providers swap models per-request; local in-process
engines need a process restart).

### Running it

```bash
# One-time install on the middleware host (the client in our setup):
pip install -e ".[web]"

# Pick a token (treat it like an API key). One option:
export DSBX_WEB_TOKEN="$(openssl rand -hex 32)"

# Make sure the dsbx-host backend is up so the middleware has something to
# talk to (the client's CLI already does this with `dsbx doctor`):
make serve-py    # starts `dsbx serve --backend llamacpp-py` on dsbx-host

# Build the frontend bundle (output: frontend/build/):
make web-build

# Run the middleware (this also static-serves the bundle at /):
dsbx web --host 127.0.0.1 --port 8765 --frontend-dist frontend/build

# Open http://localhost:8765 in a browser, paste the token, log in.
```

For frontend development with hot reload, use the dev-mode launcher:

```bash
make web-dev   # runs FastAPI on :8765 and `pnpm dev` on :5173
```

### What is and isn't exposed

The middleware actively scrubs:

- ``base_url`` for any ``[remote.NAME]`` entry (the dsbx-host LAN address).
- ``api_key_env`` and all environment-resolved API keys.
- The ``secrets_env_file`` path itself.

The browser only ever receives:

- A list of opaque backend names with their ``Capabilities`` flags.
- The token-level outputs of inspect / generate / spec / manual.
- ``/api/v1/probe`` results (status strings only, no keys).
- For remote hosts, the model catalogue (GGUF *filenames*/paths or HF
  ids) and live slot state -- enough to drive the reload picker, but
  never the host's address.

The no-secrets-leak invariant is enforced by `tests/test_web_info.py`,
which scans every ``/api/v1/info`` payload for known sentinel values.
