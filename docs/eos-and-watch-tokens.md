# EOS, stopping, and watch tokens

This document covers how `dsbx` renders tokens, how it detects and
honours end-of-text (EOS), and the `--watch` family of flags for tracing the
probability of specific tokens across a fixed context.

## Token rendering

Tokens that look identical in a column actually differ by leading/trailing
whitespace -- `"I"`, `" I"` and `"I "` are three different ids. The renderer
surfaces this with explicit markers:

- Leading / trailing spaces -> `␣` (one per space), so ` I` reads as `␣I` and
  `I ` reads as `I␣`. Internal spaces in prose stay untouched.
- Newline -> `↵`, tab -> `→`, other control bytes -> `\xNN`.
- Empty token -> `<empty>` (dim).
- Special tokens (EOS/BOS/PAD, `<|im_start|>`, `<|endoftext|>`, anything the
  tokenizer marks special or any text matching `<|...|>`) render in magenta
  bold.

## EOS detection and stopping

Each backend reports its end-of-text token ids in
`Capabilities.eos_token_ids`:

- **HF transformers**: read from `model.config.eos_token_id` and
  `tokenizer.eos_token_id` (both, deduped -- modern models like Qwen list
  several for chat templates).
- **`llamacpp-py`**: `Llama.token_eos()` plus `Llama.token_eot()` when the
  binding exposes it (Qwen-style chat templates).
- **`llamacpp` HTTP** and **OpenAI-compat providers**: the server API doesn't
  expose this, so EOS detection is unavailable there.

`dsbx generate` honours EOS by default: the moment the sampler picks an EOS id,
the loop stops and the footer reads
`stopped on EOS: model emitted <|endoftext|> (id=...)`. The footer also reports
`stopped on --stop token: ...` and
`reached --max-tokens=... (model did not emit EOS).`, so it's always clear *why*
generation halted. Set `respect_eos=False` at the engine level to probe what the
model would emit past EOS.

On the OpenAI-compat path `respect_eos=False` used to be a silent no-op (no
documented field to disable EOS halting server-side). That's now fixed for
providers that opt into the Fireworks-style `ignore_eos` field: when
`ProviderConfig.supports_ignore_eos` is true the request ships `ignore_eos: true`
and the model actually keeps emitting past its EOS. The browser's `respect EOS`
checkbox is enabled on those backends. Providers without the flag (NIM,
OpenRouter, LM Studio chat-only) get an advisory note on the `usage` SSE frame,
so the UI can say "the cloud ignored this flag" instead of pretending.

`dsbx inspect` (and `:caps` inside a session) prints the configured EOS ids in
the banner, e.g. `EOS ids: 248044=<special>`. The magenta marker means the
token's printable form is empty (a true control token). Typing `<|endoftext|>`
literally in your prompt does **not** tokenize as the EOS id -- the binding's
`tokenize()` BPE-encodes the literal characters; the real EOS id only appears
when the model itself chooses it.

## Recipe: track P(EOS) across a fixed context

Because EOS often detokenizes to empty/unprintable text, `--watch ' Paris'`
can't reach it -- there's no string to type. Use the id-based watches instead:

```bash
# Pull every EOS id from capabilities and add a column per id.
dsbx inspect --watch-eos 'The weather today is surprisingly dry.'

# Or pin a specific id (any reserved/control token, not just EOS):
dsbx inspect --watch-id 248044 --watch-id 151643 'A short test.'

# Mix freely with text watches:
dsbx inspect --watch ' Paris' --watch-eos --watch-id 1234 "France's capital is"
```

What you get:

- One extra column per watched id; the header reads e.g. `watch EOS:248044` or
  `watch id=1234 ' Paris'` (the piece is appended for sanity).
- Each row shows the **exact** logprob + rank for that id at that position --
  even on full-vocab backends, even if the id falls outside the top-k for that
  step (`watch_ids` queries the distribution directly).
- A trailing **predict-next** row, marked `N (next)` in the position column,
  shows what the model would emit *after* the entire prompt. For a prompt
  ending in a period, the EOS column on the `(next)` row tells you how strongly
  the model "wants to finish".
- Duplicates across `--watch` / `--watch-id` / `--watch-eos` are deduped by id.
- Backends without EOS info (HTTP llama.cpp, cloud providers) print a yellow
  warning and the `--watch-eos` column is simply omitted.

Inside a `dsbx session` the same flags work after the `inspect` keyword, with
the model staying loaded between queries so you can scan many prompts cheaply.

## A note on colours over SSH

The TUI runs **locally** on the client and talks to the GPU host over HTTP, so
`rich`'s TTY detection works correctly and colour rendering is no problem. The
`--color always|never` / `FORCE_COLOR` / `ssh -t` workarounds are still in the
binary (and documented in `dsbx --help`) for the rare case where someone insists
on `ssh host 'dsbx inspect ...'`, but that path is no longer recommended.
