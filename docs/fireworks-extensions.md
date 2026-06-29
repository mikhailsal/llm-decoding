# Fireworks `/v1/completions` extensions

Fireworks exposes a small zoo of extension fields on `/v1/completions` that turn
the sandbox into a much better learning and debugging tool. They are wired
through `ProviderConfig.supports_*` flags so other providers stay on the
conservative OpenAI-compatible subset. This is the most "frontier-provider
internals" part of the project, which is why it lives in its own document.

## Field-by-field

- **`ignore_eos`** -- unlocks the `respect EOS` checkbox; the model keeps
  emitting past its EOS id. See [EOS, stopping, and watch tokens](eos-and-watch-tokens.md).
- **`perf_metrics_in_response`** -- always on. The provider returns a
  `perf_metrics` block (TTFT, prefill/generation durations, cached prompt
  tokens, backend host) that the middleware surfaces as a dedicated `perf` SSE
  frame; the browser renders it as a *server timings* panel next to the running
  completion.
- **`service_tier`** -- per-request selector (`default` / `priority`). The UI
  shows the dropdown only when `Capabilities.supports_service_tier` is true.
- **`prompt_cache_key`** -- requests sharing the same key route to the same
  backend replica to maximize KV-cache hit rate (great for manual decoding,
  where the prefix is stable).
- **`x-session-affinity` + `x-multi-turn-session-id`** (HTTP headers) -- when
  `session_id` is set on a request these go on the wire; the second enables
  Fireworks' **MoE Router Replay (R3)**, making MoE expert routing deterministic
  across the turns of a multi-step session.

## How the support landed (phased)

The Fireworks integration was built in phases; the field set above is the
union of all of them.

**Phases 1-2** -- wire mechanics + expanded sampling (`typical_p`, `mirostat`,
repetition / frequency / presence penalties).

**Phase 3** -- swapped the cloud parser onto the **NewLogProbs** format
(`logprobs: true` + `top_logprobs: N` instead of a single integer):

- Token candidates now carry the **real model `token_id`** straight from the
  response, instead of the synthetic interned ids the legacy parser had to
  invent. That makes `--watch-id N` finally produce meaningful "what's the
  probability of token 1234?" traces against cloud providers.
- When `Capabilities.supports_sampling_mask` is true, the request also sets
  `sampling_mask: 'count'` and the response carries `sampling_mask_count` per
  position -- the number of tokens that survived the server's sampling filter
  stack. Both `/generate` and `/inspect` render this as an **eligible** column
  right after the probability bar.

**Phase 4** -- the remaining Fireworks-only knobs:

- **`raw_output: true`** (always on for Fireworks). The provider's diagnostics
  block (`prompt_fragments`, `prompt_token_ids`, `grammar`, ...) flows through a
  dedicated `raw_output` SSE frame and renders as a "what the model saw" panel.
  Most useful when a custom chat template silently ate your system prompt -- you
  see it directly instead of guessing.
- **`logit_bias`** -- a field on `GenerateRequest`. The generate page exposes a
  row editor (token_id + bias in [-100, 100]) gated on
  `Capabilities.supports_logit_bias`. Wire shape matches OpenAI
  (`{"<token_id>": float}`); invalid rows are dropped silently on both client
  and backend, so a single typo never fails the whole request. Use cases: ban a
  token (`-100`), nudge a rare option past a tight `top_p` (`+5..+15`), force a
  token in a grammar-constrained setup (`+100`).

**Phase 5** -- collapses the "include prompt logits" workflow from two network
round trips into one when the backend advertises `supports_combined_echo_stream`:

- A single `echo=true` + `stream=true` request returns BOTH the echoed
  per-prompt-token logprobs AND the streamed generated tokens on one
  connection. The frontend wire shape is unchanged
  (`prompt_score? -> step* -> perf? -> raw_output? -> usage -> done`), so
  existing consumers keep working.
- **`echo_last`** (Fireworks-specific, gated on `supports_echo_last`) restricts
  the echoed positions to the last N prompt tokens -- handy for long prompts
  where you only care about the trailing context. The generate page surfaces it
  as a small "echo last N" knob.
- `scripts/smoke_fireworks_echo_stream.py` is a one-shot diagnostic that POSTs
  an `echo + stream + max_tokens>0 + logprobs` body against the real Fireworks
  API and prints the chunk-by-chunk ordering, so you can confirm a new
  deployment tolerates the combo before flipping `supports_combined_echo_stream`
  on in `config.toml`.
