<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import BackendSelect from '$lib/components/BackendSelect.svelte';
  import ModelInput from '$lib/components/ModelInput.svelte';
  import RemoteModelControl from '$lib/components/RemoteModelControl.svelte';
  import CapabilityBadges from '$lib/components/CapabilityBadges.svelte';
  import ChipInput from '$lib/components/ChipInput.svelte';
  import ConfidenceBar from '$lib/components/ConfidenceBar.svelte';
  import TokenText from '$lib/components/TokenText.svelte';
  import TokenInline from '$lib/components/TokenInline.svelte';
  import CompletionToken from '$lib/components/CompletionToken.svelte';
  import TokenComposer from '$lib/components/TokenComposer.svelte';
  import Toast from '$lib/components/Toast.svelte';
  import { apiStream } from '$lib/api';
  import { info } from '$lib/stores/info';
  import { probFromLogprob, tokenBackgroundClass, formatProbPct } from '$lib/render';
  import type {
    GenStep,
    StepResult,
    TokenCandidate,
    Watched,
    BackendInfo,
    UsagePayload,
    PerfMetricsPayload,
    RawOutputPayload
  } from '$lib/types';

  type Mode = 'inspect' | 'generate' | 'manual';
  let lastMode = $state<Mode>('generate');

  let backend = $state<string>('');
  let model = $state<string>('');
  let prompt = $state('Once upon a time');
  let maxTokens = $state(20);
  // ``alternatives`` is the number of top-k logprobs we both FETCH from
  // the backend AND show in the table. There was a separate "alts shown"
  // knob, but offering both invited a confusing failure mode: setting
  // alts > fetched silently rendered empty rows, and setting fetched
  // beyond ``capabilities.max_top_logprobs`` was clamped server-side
  // without telling the user. One knob, one truth -- and we clamp the
  // input's ``max`` attribute to the live backend's reported ceiling.
  let alternatives = $state(8);
  let seed = $state(0);
  let stopTexts = $state<string[]>([]);
  let stopIds = $state<string[]>([]);
  let respectEos = $state(true);
  // Defaults to true in the unified Decode workbench: showing per-prompt-
  // token logits is the highest-signal way to introduce the user to the
  // model's distribution, and on Fireworks combined-echo+stream means we
  // get it back in ONE upstream request (zero added RPS cost). The user
  // can still untick to skip the prompt-score frame.
  let includePrompt = $state(true);
  // Watch chips panel (ported from the old /inspect page): every request
  // ships these so the watched columns appear in both prompt-logits and
  // generation-steps tables. Strings during edit, server-side resolved
  // to ids in ``_resolve_watches``.
  let watchTexts = $state<string[]>([]);
  let watchIds = $state<string[]>([]);
  // Token ids to splice in BEFORE the user's prompt -- the
  // "prepend tokens" workflow. Same string-of-ints shape as ``watchIds``
  // so it can use the existing ChipInput component; the request body
  // assembler parses to numbers and drops anything non-finite. Most
  // users will populate this via the "fill BOS" helper next to the
  // chip-input; advanced users can type any id to see how the model
  // conditions on it.
  let prependTokenIds = $state<string[]>([]);
  // Snapshot of how many prepend ids were actually sent for the last
  // generate/inspect/manual call. Read by the prompt-logits table to
  // mark which row carries the BOS-conditioned first-real-prompt-token
  // prediction (row at position == prependCount). We pin the count at
  // request time rather than reading ``prependTokenIds.length``
  // directly so the highlight survives the user editing the chip
  // between requests -- the rendered table reflects the run that
  // produced it.
  let lastRunPrependCount = $state<number>(0);
  let watchEos = $state(false);
  // ``echo_last=N`` (Fireworks combined echo+stream path only): echo
  // logprobs for the LAST N prompt tokens instead of every position.
  // 0 = echo the whole prompt (the default). Only meaningful when
  // ``includePrompt`` is on AND the backend can do combined
  // echo+stream; otherwise the field is hidden.
  let echoLast = $state(0);
  let showMarkers = $state(true);
  // Provider-extension knobs. ``serviceTier`` defaults to "default" (the
  // standard serverless tier); switching to "priority" is opt-in and only
  // matters when the active backend advertises supports_service_tier.
  let serviceTier = $state<'default' | 'priority'>('default');

  type SamplerName =
    | 'greedy'
    | 'temperature'
    | 'top_k'
    | 'top_p'
    | 'min_p'
    | 'typical'
    | 'mirostat';
  let sampler = $state<SamplerName>('greedy');
  let temperature = $state(1.0);
  let samplerTopK = $state(40);
  let topP = $state(0.9);
  let minP = $state(0.05);
  let typicalP = $state(0.95);
  let mirostatTarget = $state(5.0);
  let mirostatLr = $state(0.1);
  // Penalties ride along on every sampler. Defaults are no-ops so they
  // disappear from the wire body unless the user explicitly opts in.
  let repetitionPenalty = $state(1.0);
  let frequencyPenalty = $state(0.0);
  let presencePenalty = $state(0.0);

  // ``promptSteps`` is the per-prompt-token distribution returned by the
  // optional ``prompt_score`` SSE frame; rendered as extra rows above the
  // generation steps. Keeping it separate from ``steps`` makes the table
  // alignment trivial (one column for "prompt token" vs "step number").
  let promptSteps = $state<StepResult[]>([]);
  let promptNote = $state<string>('');
  let steps = $state<GenStep[]>([]);
  let stopReason = $state<string | null>(null);
  let streamError = $state<string | null>(null);
  let busy = $state(false);
  let cancelFn: (() => void) | null = null;
  // Per-run usage badge under "running completion": shows how many HTTP
  // requests this run consumed (so the historical "20 requests for 20
  // tokens" pattern is visible at a glance) plus prompt/completion
  // token totals reported by the provider (or computed locally for
  // non-cloud backends). Cleared at the start of each new run.
  let usage = $state<UsagePayload | null>(null);
  let perf = $state<PerfMetricsPayload | null>(null);
  // ``rawOutput`` is the Fireworks "what the model actually saw"
  // diagnostics block (prompt_fragments / prompt_token_ids / grammar
  // / ...). Rendered as a dedicated panel below "server timings" when
  // the backend advertises ``supports_raw_output`` and the run
  // produced a non-empty payload.
  let rawOutput = $state<RawOutputPayload | null>(null);
  // ``logit_bias`` editor: a list of (token_id, bias) rows the user
  // can add / delete. We keep them as strings while editing so the
  // user can type freely (a partial "-1" before the "0" doesn't snap
  // back to "0" mid-typing) and parse + clamp at submit time.
  type LogitBiasRow = { id: string; tokenId: string; bias: string };
  let logitBiasRows = $state<LogitBiasRow[]>([]);

  // -------- Manual decoding (browser-side ephemeral state) ----------
  // The unified workbench's "Manual" button activates an inline picker
  // that DOES NOT touch the deleted manual-sessions backend. State lives
  // entirely in the browser: ``pickedIds`` grows as the user picks
  // candidates, and each pick fires a fresh ``/generate/stream`` call
  // with ``prefix_token_ids = pickedIds`` so the model sees
  // ``tokenize(prompt) + pickedIds`` as one continuous sequence. The
  // ``manualSessionId`` + ``manualCacheKey`` UUIDs are auto-generated on
  // enter and stay stable for the lifetime of the manual session, which
  // on Fireworks reuses the KV cache + MoE expert routing across picks
  // (essentially free per-pick after the first).
  let manualMode = $state(false);
  let manualSessionId = $state<string>('');
  let manualCacheKey = $state<string>('');
  let pickedIds = $state<number[]>([]);
  let pickedTexts = $state<string[]>([]);
  let pickedProbs = $state<(number | null)[]>([]);
  // Current picker distribution -- the StepResult emitted by the most
  // recent generate-stream call. The picker table reads ``candidates``
  // and ``watched`` from it; ``chosen`` is unused in manual mode.
  let manualDistribution = $state<StepResult | null>(null);
  let forceText = $state('');
  let forceId = $state('');

  // ``tokenCache`` is an id->text dictionary populated from every
  // ``prompt_score`` and ``step`` SSE frame across the lifetime of
  // the page. It powers two UX wins:
  //
  //   1. Tokens rendered in the prompt-logits and generation-steps
  //      tables become clickable. One tap adds the token (with a
  //      sensible default bias) to the ``logit bias`` editor, so the
  //      user doesn't have to copy the numeric id by hand.
  //   2. The ``token_id`` input inside the bias editor is backed by
  //      an HTML ``<datalist>`` whose options are every cached
  //      (id, text) pair. Typing ``" Pa"`` autocompletes to the
  //      Paris token id (etc.), making manual entry usable.
  //
  // The cache only grows for the active session and is wiped on a
  // full page reload, which is fine -- token vocabularies differ per
  // backend and persisting them across reloads invites confusing
  // suggestions when the user swaps backends mid-flow.
  let tokenCache = $state<Record<number, string>>({});

  function rememberToken(id: number | null | undefined, text: string): void {
    if (id === null || id === undefined) return;
    if (!Number.isFinite(id)) return;
    // Synthetic intern ids (>= 1<<24) come from local backends that
    // can't expose real model ids; biasing on them would be a no-op
    // upstream because the cloud /completions endpoint never sees
    // them. Skip caching synthetics to keep the suggestion list
    // useful.
    if (id >= 1 << 24) return;
    if (tokenCache[id] !== undefined) return;
    tokenCache = { ...tokenCache, [id]: text };
  }

  function rememberFromStep(s: StepResult | null | undefined): void {
    if (!s) return;
    if (s.chosen) rememberToken(s.chosen.token_id, s.chosen.text);
    for (const c of s.candidates ?? []) rememberToken(c.token_id, c.text);
  }

  /**
   * Add a token to the logit_bias editor. If a row for this id
   * already exists we don't add a duplicate -- we just leave the
   * existing row in place (the user can edit its bias). Default
   * bias of -100 mirrors the Fireworks "ban this token" idiom; the
   * user can flip the sign in the row's bias input if they wanted
   * to nudge UP instead.
   */
  function addToBias(tokenId: number, defaultBias: number = -100): void {
    const tidStr = String(tokenId);
    if (logitBiasRows.some((r) => r.tokenId === tidStr)) return;
    logitBiasRows = [
      ...logitBiasRows,
      { id: crypto.randomUUID(), tokenId: tidStr, bias: String(defaultBias) }
    ];
  }

  // -------- Token-click actions (running completion + tables) ---------
  // These power the unified ``CompletionToken`` menu so a click does the
  // same thing on EVERY backend (the old behaviour gated all token
  // clicks on logit_bias support, leaving dsbx-host-py tokens inert while
  // fireworks tokens were clickable).

  /** Pin a token to the watch panel so its per-step probability shows up
   *  as a dedicated column in the tables below. */
  function addToWatchToken(tokenId: number, text: string): void {
    rememberToken(tokenId, text);
    const s = String(tokenId);
    if (!watchIds.includes(s)) watchIds = [...watchIds, s];
  }

  /** Append a single token's text to the prompt (the per-token sibling of
   *  the "move generation to prompt" button). */
  function addTokenToPrompt(text: string): void {
    prompt = prompt + text;
  }

  // Snapshot of the prompt text that produced the on-screen generation.
  // Captured at run start so the running-completion view renders a STABLE
  // prefix (the live ``prompt`` may be edited afterwards, and "move to
  // prompt" rewrites it -- using the snapshot keeps the displayed
  // prefix+steps consistent and stops "move to prompt" from duplicating
  // the completion back into its own prefix).
  let runPromptText = $state('');

  /** Fold the run's prompt + everything the model generated into the
   *  prompt box as a single continuous string, so the user can keep
   *  generating from exactly what the running-completion shows. Replaces
   *  (rather than appends) so an edited prompt never desyncs the result. */
  function moveCompletionToPrompt(): void {
    prompt = runPromptText + completionText;
  }

  /** Scroll the matching generation-steps row into view and flash it, so
   *  "find in list" from the running completion lands the user on the
   *  exact step row. */
  function findStepInList(step: number): void {
    const el = document.getElementById(`gen-step-${step}`);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.classList.add('row-flash');
    window.setTimeout(() => el.classList.remove('row-flash'), 1600);
  }

  // ``completionText`` is everything the model has emitted this run (the
  // concatenated per-step token text). Drives the "move to prompt"
  // button so the user can fold a generation back into the prompt and
  // keep going.
  let completionText = $derived<string>(
    steps.map((s) => s.decision.token_text).join('')
  );

  // Prefix shown in the running-completion view. Once a run has produced
  // output we show the captured ``runPromptText`` snapshot (NOT the live
  // ``prompt``, which the user may edit or which "move to prompt" rewrites
  // -- either would otherwise duplicate/desync the displayed completion).
  // Before any run we mirror the live prompt so the box isn't empty.
  let runningPromptDisplay = $derived<string>(
    steps.length > 0 || promptSteps.length > 0 || pickedTexts.length > 0
      ? runPromptText
      : prompt
  );

  /** Reset the generation view (steps / tables / usage / manual session).
   *  Called when the user switches backend or model so stale output never
   *  looks like it came from the newly-selected model/provider. */
  function clearRun(): void {
    steps = [];
    promptSteps = [];
    promptNote = '';
    stopReason = null;
    streamError = null;
    usage = null;
    perf = null;
    rawOutput = null;
    manualDistribution = null;
    manualMode = false;
    pickedIds = [];
    pickedTexts = [];
    pickedProbs = [];
    runPromptText = '';
  }

  // ``producedKey`` records the backend+model that generated whatever is
  // currently on screen (set at the start of every run). Clear-on-switch
  // compares against it so we only wipe the view when the selection
  // actually diverges from what produced the displayed output.
  let producedKey: string | null = null;

  // Clear-on-switch: changing the active backend or model wipes the view
  // so stale output never looks like it came from the newly-selected
  // model/provider. Triggered from explicit change handlers
  // (``onBackendChange``, ``ModelInput``'s ``onChange``, and the remote
  // reload's ``onReady``) rather than a reactive effect, so there is no
  // read/write cycle to trip Svelte's update-depth guard. An in-flight
  // stream is cancelled first.
  function maybeClearForSwitch(): void {
    const hasOutput =
      steps.length > 0 || promptSteps.length > 0 || pickedIds.length > 0 || manualMode;
    if (!hasOutput) return;
    if (`${backend}\u0000${model}` === producedKey) return;
    cancelFn?.();
    busy = false;
    clearRun();
  }

  // Cached sorted view for the <datalist>: numeric ascending so the
  // dropdown reads predictably (the user picks IDs by text 90% of
  // the time anyway -- the alphabetical-by-text option would shuffle
  // every time the cache grew, which is jarring).
  let tokenCacheEntries = $derived(
    Object.entries(tokenCache)
      .map(([id, text]) => ({ id: Number(id), text }))
      .sort((a, b) => a.id - b.id)
  );

  let backendInfo = $derived<BackendInfo | null>(
    $info.info?.backends.find((b) => b.name === backend) ?? null
  );

  // Resolve the capabilities envelope for the CURRENTLY-SELECTED model.
  // Cloud providers carry a per-model map (``models_caps``) so
  // model-specific quirks reach the UI without leaking between models:
  // e.g. picking glm-5p1 shouldn't inherit gpt-oss-120b's BOS, and
  // gpt-oss-20b should NOT pretend it supports ``sampling_mask``
  // (the upstream returns HTTP 400 for that field). For remote /
  // local backends ``models_caps`` is empty and the fallback to
  // ``backendInfo.capabilities`` keeps behaviour identical to before.
  let activeCaps = $derived<BackendInfo['capabilities'] | null>(
    (model ? (backendInfo?.models_caps?.[model] ?? null) : null)
      ?? backendInfo?.capabilities
      ?? null
  );

  // ``respect EOS`` is locked only for cloud backends that DON'T advertise
  // the Fireworks-style ``ignore_eos`` field. Fireworks unlocks this
  // checkbox (we ship ``ignore_eos: true`` on the wire); NIM, OpenRouter,
  // and LM Studio still leave it pinned because they have no documented
  // escape hatch -- the upstream silently halts on EOS no matter what.
  let respectEosLocked = $derived<boolean>(
    backendInfo?.family === 'cloud' && !activeCaps?.supports_ignore_eos
  );
  $effect(() => {
    // Force ``respect EOS`` on when the active backend locks it. Track
    // only ``respectEosLocked``; untrack the read+write of ``respectEos``
    // so the self-write (and the checkbox ``bind:checked`` write-back)
    // can't re-trigger this effect into the update-depth guard.
    const locked = respectEosLocked;
    untrack(() => {
      if (locked && !respectEos) respectEos = true;
    });
  });
  // Show the service-tier selector only on backends that advertise it
  // (Fireworks). Other backends silently ignore the field anyway, but
  // showing a knob that does nothing is bad UX.
  let serviceTierSupported = $derived<boolean>(
    !!activeCaps?.supports_service_tier
  );
  // Server-timings panel visibility tracks supports_perf_metrics; we only
  // render it when (a) the backend can report metrics and (b) we actually
  // received a non-empty perf frame in the last run.
  let perfPanelSupported = $derived<boolean>(
    !!activeCaps?.supports_perf_metrics
  );
  // ``sampling_mask=count`` tells us how many tokens survived
  // server-side sampling filters at each position. When the backend
  // advertises this capability we surface a dedicated "eligible after
  // filters" column in both prompt-logits and generation-steps tables;
  // otherwise the column would just render "?" everywhere which is
  // worse than not showing it at all.
  let samplingMaskSupported = $derived<boolean>(
    !!activeCaps?.supports_sampling_mask
  );
  let rawOutputSupported = $derived<boolean>(
    !!activeCaps?.supports_raw_output
  );
  let logitBiasSupported = $derived<boolean>(
    !!activeCaps?.supports_logit_bias
  );
  let combinedEchoStreamSupported = $derived<boolean>(
    !!activeCaps?.supports_combined_echo_stream
  );
  // Backends that can accept ``prepend_token_ids`` on score_prompt /
  // generate -- i.e. those that tokenize locally and can splice extra
  // ids into the sequence at the TOKEN level. Cloud providers
  // (Fireworks today) tokenize ``prompt: str`` server-side and can't
  // safely inject extra ids; the UI's "fill BOS" helper and the
  // prepend chip-input gate on this flag so a click never goes through
  // a path where it would silently no-op.
  let prependSupported = $derived<boolean>(
    !!activeCaps?.supports_prepend_token_ids
  );
  // Whether the backend exposes a real local tokenizer through
  // ``/api/v1/tokenize`` (real ids per word piece, not the synthetic
  // single-id stub). Used to gate the live token preview under the
  // prompt textarea: without a real tokenizer the preview would
  // teach the wrong thing (one chip for the whole prompt), so we
  // simply hide the component instead. Cloud backends flip this on
  // once their per-model HF tokenizer has been fetched + cached
  // (Fireworks gpt-oss / Qwen out of the box; Llama once HF_TOKEN
  // grants access to the gated repo).
  let localTokenizeSupported = $derived<boolean>(
    !!activeCaps?.supports_local_tokenize
  );
  // Empty-prompt guard. An autoregressive model computes P(next | prior
  // tokens); with ZERO input tokens there is literally nothing to
  // condition the first prediction on, so a run can't even begin (and
  // every provider rejects / no-ops an empty prompt). Rather than fire a
  // request that silently does nothing, we disable the run buttons and
  // explain why -- and point at the fix (type text, or insert a special
  // token like BOS from the palette to give the model a starting point).
  let promptEmpty = $derived<boolean>(prompt.length === 0);
  const emptyPromptNote =
    'Empty prompt — nothing to run. An autoregressive model predicts the ' +
    'next token from the tokens before it; with zero input tokens there is ' +
    'nothing to condition on. Type some text, or insert a special token ' +
    '(e.g. the model’s BOS) from the palette to seed generation.';
  // Chat-only providers (NIM / OpenRouter) are registered but inert
  // until proper chat-mode UI lands; the middleware rejects
  // generate-stream requests against them with a 400. We mirror the
  // gate here so the button is visibly disabled and carries the
  // backend's ``notes`` as a tooltip explanation, instead of letting
  // the click round-trip and bounce off a 400.
  let generationDisabled = $derived<boolean>(
    !!activeCaps?.generation_disabled
  );
  let generationDisabledNote = $derived<string>(
    activeCaps?.notes ?? ''
  );

  // ``alternatives`` ceiling comes from the backend's capabilities. Cloud
  // providers cap aggressively (Fireworks: 5, NIM/OpenRouter: 20,
  // LM Studio: 10); local backends with a real vocab cap much higher
  // (we use 200 as a sane upper bound so the input doesn't become a
  // free-form free-for-all). Without the cap the user could ask for
  // ``top_k=100`` against Fireworks and silently get 5 back -- the
  // exact "silently ignored" UX the user pointed at.
  let altsMax = $derived<number>(activeCaps?.max_top_logprobs ?? 50);
  $effect(() => {
    // Re-clamp ONLY when the cap changes (backend swap / capabilities
    // refresh). ``untrack`` the read+write of ``alternatives`` so this
    // effect never depends on -- nor re-triggers from -- its own write;
    // pairing a self-writing effect with the number input's
    // ``bind:value`` write-back otherwise spins the update-depth guard.
    // Never grow the value automatically: if the user picked 3 and the
    // cap is 20, leave it at 3.
    const cap = altsMax;
    untrack(() => {
      if (alternatives > cap) alternatives = cap;
      if (alternatives < 1) alternatives = 1;
    });
  });

  // ``seed`` only changes the result when the sampler has randomness to
  // seed -- greedy is deterministic by definition (argmax of the
  // distribution), so the seed is a pure no-op there. Rather than let
  // a user think they're varying the run by editing seed under greedy,
  // we lock the input and explain why.
  let seedLocked = $derived<boolean>(sampler === 'greedy');

  // ``include_prompt`` is most useful when the backend supports
  // per-prompt-token logprobs (HF, llamacpp_py, Fireworks/echo). On
  // chat-only providers (NIM, OpenRouter, LM Studio chat-only) it
  // still works but degrades to a single "what comes next?" row --
  // we leave the toggle enabled but flag the degraded mode.
  let promptScoreDegraded = $derived<boolean>(
    activeCaps?.prompt_logprobs === false
  );

  onMount(async () => {
    if (!$info.info) await info.refresh();
    backend = $info.info?.default_backend ?? '';
    model = backendInfo?.loaded_model ?? '';
  });

  // When the user picks a new backend, reset ``model`` to the new backend's
  // default so we don't accidentally pass an incompatible name on. The user
  // can edit it back if they want.
  function onBackendChange(next: string) {
    info.select(next);
    const b = $info.info?.backends.find((x) => x.name === next) ?? null;
    model = b?.loaded_model ?? '';
    maybeClearForSwitch();
  }

  function samplerParams(): Record<string, number | null> {
    const penalties: Record<string, number> = {};
    if (repetitionPenalty !== 1.0) penalties.repetition_penalty = repetitionPenalty;
    if (frequencyPenalty !== 0.0) penalties.frequency_penalty = frequencyPenalty;
    if (presencePenalty !== 0.0) penalties.presence_penalty = presencePenalty;
    switch (sampler) {
      case 'greedy':
        return { ...penalties };
      case 'temperature':
        return { temperature, ...penalties };
      case 'top_k':
        return { temperature, top_k: samplerTopK, ...penalties };
      case 'top_p':
        return { temperature, top_p: topP, ...penalties };
      case 'min_p':
        return { temperature, min_p: minP, ...penalties };
      case 'mirostat':
        return {
          temperature,
          mirostat_target: mirostatTarget,
          mirostat_lr: mirostatLr,
          ...penalties
        };
      case 'typical':
        return { temperature, typical_p: typicalP, ...penalties };
    }
  }

  /**
   * Convert the editor rows to the wire shape ``{<token_id>: <bias>}``
   * with string keys (JSON requirement). Silently drop rows with
   * unparseable ids or out-of-range biases; the backend would reject
   * those anyway and we'd rather the user keep their partial input
   * visible than have the page nuke half their edits at submit time.
   * Returns ``undefined`` -- not an empty object -- when there are no
   * valid rows, so the request omits the field entirely.
   */
  function collectLogitBias(): Record<string, number> | undefined {
    if (!logitBiasSupported) return undefined;
    const out: Record<string, number> = {};
    for (const row of logitBiasRows) {
      const tid = Number.parseInt(row.tokenId, 10);
      const bias = Number.parseFloat(row.bias);
      if (!Number.isFinite(tid) || !Number.isFinite(bias)) continue;
      if (bias < -100 || bias > 100) continue;
      out[String(tid)] = bias;
    }
    return Object.keys(out).length ? out : undefined;
  }

  /**
   * Build the request body for a generate-stream call. Mode-specific
   * fields (max_tokens, include_prompt, prefix_token_ids, manual session
   * UUIDs) are layered on top of the common fields here so the three
   * action buttons stay one-liners. ``prefix`` defaults to the manual
   * picker's accumulated picks when set; pass ``[]`` for inspect/generate.
   */
  function buildRequest(opts: {
    maxTokensOverride?: number;
    includePromptOverride?: boolean;
    prefix?: number[];
    forManual?: boolean;
  }): Record<string, unknown> {
    const stop_ids = stopIds
      .map((s) => Number.parseInt(s, 10))
      .filter((n) => Number.isFinite(n));
    const watch_ids_resolved = watchIds
      .map((s) => Number.parseInt(s, 10))
      .filter((n) => Number.isFinite(n));
    const includeP = opts.includePromptOverride ?? includePrompt;
    return {
      backend,
      model: model || undefined,
      prompt,
      sampler: { name: sampler, params: samplerParams() },
      max_tokens: opts.maxTokensOverride ?? maxTokens,
      top_k: alternatives,
      stop_texts: stopTexts,
      stop_ids,
      seed,
      respect_eos: respectEos,
      include_prompt: includeP,
      service_tier: serviceTierSupported ? serviceTier : undefined,
      logit_bias: collectLogitBias(),
      echo_last:
        includeP && combinedEchoStreamSupported && echoLast > 0 ? echoLast : undefined,
      watch_texts: watchTexts,
      watch_ids: watch_ids_resolved,
      watch_eos: watchEos,
      prefix_token_ids: opts.prefix ?? [],
      prepend_token_ids: (() => {
        const ids = prependSupported
          ? prependTokenIds
              .map((s) => Number.parseInt(s, 10))
              .filter((n) => Number.isFinite(n))
          : [];
        // Pinned at request time so the prompt-logits table can
        // highlight the "BOS-conditioned" row even after the user
        // edits the chip-input between runs (the rendered table
        // reflects the run that produced it, not the current input).
        lastRunPrependCount = ids.length;
        return ids;
      })(),
      // Manual mode pins both UUIDs so Fireworks can reuse the KV cache
      // and MoE expert routing across picks; for the other modes we
      // leave them ``undefined`` so each click is a fresh request.
      session_id: opts.forManual ? manualSessionId : undefined,
      prompt_cache_key: opts.forManual ? manualCacheKey : undefined
    };
  }

  /**
   * Generic streaming runner shared by all three buttons. ``onStep``
   * lets the manual-mode caller intercept the single emitted step and
   * stash it into ``manualDistribution`` instead of appending to the
   * scrolling ``steps`` table.
   */
  async function streamRun(
    body: Record<string, unknown>,
    opts: {
      resetUi: boolean;
      onStep?: (gs: GenStep) => void;
      onPromptScore?: (steps: StepResult[], note: string) => void;
    }
  ): Promise<void> {
    streamError = null;
    stopReason = null;
    // Stamp the config that produces this output so a later backend/model
    // switch knows whether the displayed result is stale.
    producedKey = `${backend}\u0000${model}`;
    runPromptText = prompt;
    if (opts.resetUi) {
      steps = [];
      promptSteps = [];
      promptNote = '';
      usage = null;
      perf = null;
      rawOutput = null;
    }
    busy = true;
    const stream = apiStream('/api/v1/generate/stream', body);
    cancelFn = stream.cancel;
    try {
      for await (const evt of stream.events) {
        if (evt.event === 'step') {
          const gs = evt.step as GenStep;
          rememberFromStep(gs.step_result);
          rememberToken(gs.decision?.token_id, gs.decision?.token_text ?? '');
          if (opts.onStep) {
            opts.onStep(gs);
          } else {
            steps = [...steps, gs];
          }
        } else if (evt.event === 'prompt_score') {
          const ps = (evt as { steps: StepResult[] }).steps ?? [];
          const note = (evt as { note?: string }).note ?? '';
          for (const s of ps) rememberFromStep(s);
          if (opts.onPromptScore) {
            opts.onPromptScore(ps, note);
          } else {
            promptSteps = ps;
            promptNote = note;
          }
        } else if (evt.event === 'perf') {
          const p = (evt as { metrics?: PerfMetricsPayload }).metrics;
          perf = p && typeof p === 'object' ? p : null;
        } else if (evt.event === 'raw_output') {
          const p = (evt as { payload?: RawOutputPayload }).payload;
          rawOutput = p && typeof p === 'object' ? p : null;
        } else if (evt.event === 'usage') {
          const u = evt as unknown as UsagePayload & { event: 'usage' };
          usage = {
            requests: u.requests ?? 0,
            prompt_tokens: u.prompt_tokens ?? null,
            completion_tokens: u.completion_tokens ?? null,
            total_tokens: u.total_tokens ?? null,
            notes: Array.isArray(u.notes) ? u.notes : []
          };
        } else if (evt.event === 'done') {
          stopReason = (evt as { stop_reason?: string | null }).stop_reason ?? null;
          const err = (evt as { error?: string | null }).error;
          if (err) streamError = err;
        }
      }
    } catch (exc) {
      streamError = exc instanceof Error ? exc.message : String(exc);
    } finally {
      busy = false;
      cancelFn = null;
    }
  }

  async function runInspect() {
    lastMode = 'inspect';
    manualMode = false;
    await streamRun(buildRequest({ maxTokensOverride: 1, includePromptOverride: true }), {
      resetUi: true
    });
  }

  async function runGenerate() {
    lastMode = 'generate';
    manualMode = false;
    await streamRun(buildRequest({}), { resetUi: true });
  }

  /**
   * Enter manual decoding mode. Generates fresh per-session UUIDs (so
   * Fireworks routes pin to one replica + reuse KV cache across picks)
   * and fires the first generate-stream call with ``prefix=[]`` so the
   * picker has an initial distribution to show.
   */
  async function enterManual() {
    lastMode = 'manual';
    manualMode = true;
    manualSessionId = crypto.randomUUID();
    manualCacheKey = crypto.randomUUID();
    pickedIds = [];
    pickedTexts = [];
    pickedProbs = [];
    manualDistribution = null;
    await fetchManualNext();
  }

  function exitManual() {
    manualMode = false;
    // Leave the rendered prompt-score + completion in place so the user
    // can switch back to inspect/generate without losing context.
  }

  /**
   * Fire one generate-stream call for manual mode. ``include_prompt`` is
   * only true on the first call (no picks yet) -- once the picker has
   * a baseline distribution we don't need the per-prompt-token rows
   * again and switching to ``include_prompt=false`` lets the Fireworks
   * cache-replay path skip the echo work too.
   */
  async function fetchManualNext() {
    const firstCall = pickedIds.length === 0;
    const body = buildRequest({
      maxTokensOverride: 1,
      includePromptOverride: firstCall,
      prefix: [...pickedIds],
      forManual: true
    });
    await streamRun(body, {
      // First call resets the UI so the prompt-logits table refreshes;
      // subsequent picks leave it alone (cache reuse).
      resetUi: firstCall,
      onStep: (gs) => {
        manualDistribution = gs.step_result;
      },
      onPromptScore: firstCall
        ? (ps, note) => {
            promptSteps = ps;
            promptNote = note;
          }
        : undefined
    });
  }

  function pickCandidate(c: TokenCandidate) {
    pickedIds = [...pickedIds, c.token_id];
    pickedTexts = [...pickedTexts, c.text];
    pickedProbs = [
      ...pickedProbs,
      c.logprob !== null ? Math.exp(c.logprob) : null
    ];
    rememberToken(c.token_id, c.text);
    void fetchManualNext();
  }

  function pickForceText() {
    const text = forceText;
    if (!text) return;
    // Tokenize on the server (we don't have the tokenizer in-browser).
    // Append every token the input expanded into so multi-token forces
    // ("Paris" -> ["Pa", "ris"]) end up as a coherent prefix.
    void (async () => {
      busy = true;
      try {
        const res = await fetch('/api/v1/tokenize', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ backend, model: model || undefined, text })
        });
        if (!res.ok) {
          streamError = `tokenize failed: HTTP ${res.status}`;
          busy = false;
          return;
        }
        const data = (await res.json()) as { ids: number[] };
        for (const tid of data.ids) {
          pickedIds = [...pickedIds, tid];
          pickedTexts = [...pickedTexts, ''];
          pickedProbs = [...pickedProbs, null];
        }
        forceText = '';
        busy = false;
        await fetchManualNext();
      } catch (exc) {
        streamError = exc instanceof Error ? exc.message : String(exc);
        busy = false;
      }
    })();
  }

  function pickForceId() {
    const tid = Number.parseInt(forceId, 10);
    if (!Number.isFinite(tid)) return;
    pickedIds = [...pickedIds, tid];
    pickedTexts = [...pickedTexts, tokenCache[tid] ?? ''];
    pickedProbs = [...pickedProbs, null];
    forceId = '';
    void fetchManualNext();
  }

  function manualUndo() {
    if (pickedIds.length === 0) return;
    pickedIds = pickedIds.slice(0, -1);
    pickedTexts = pickedTexts.slice(0, -1);
    pickedProbs = pickedProbs.slice(0, -1);
    void fetchManualNext();
  }

  function manualUndoTo(idx: number) {
    if (idx < 0 || idx >= pickedIds.length) return;
    pickedIds = pickedIds.slice(0, idx);
    pickedTexts = pickedTexts.slice(0, idx);
    pickedProbs = pickedProbs.slice(0, idx);
    void fetchManualNext();
  }

  function cancel() {
    cancelFn?.();
    busy = false;
  }

  // For each emitted step, the *chosen* candidate's linear prob, used by the
  // running-completion view to color the token background.
  function chosenProb(step: GenStep): number | null {
    const id = step.decision.token_id;
    const c = step.step_result.candidates.find((c: TokenCandidate) => c.token_id === id);
    if (!c) return null;
    return probFromLogprob(c.logprob);
  }

  function watchedById(step: StepResult, id: number): TokenCandidate | null {
    const w = step.watched.find((x: Watched) => x.token_id === id);
    return w ? w.candidate : null;
  }

  /**
   * Frontend reconstructs human-readable watch column headers from what
   * IT sent (no server-side ResolvedWatch round-trip; the inspect
   * endpoint that needed it is gone). De-duplication semantics mirror
   * the server's :func:`_resolve_watches` so the columns line up with
   * the per-step ``watched`` arrays.
   */
  type WatchColumn = { label: string; tokenId: number; source: 'text' | 'id' | 'eos' };
  let watchColumns = $derived.by<WatchColumn[]>(() => {
    const out: WatchColumn[] = [];
    const seen = new Set<number>();
    // text watches: we don't know the tokenizer in-browser, so we
    // can't pre-resolve to ids. Instead we render the label up front
    // and pair it with the watched cell by scanning each row's
    // ``watched`` list for "the first id we haven't matched yet that
    // looks like it came from a text watch". Practically: text
    // watches always appear before id+eos watches in the server-side
    // resolution order, so we render them as placeholder columns and
    // let the first N watched entries fill them in. (The plan's
    // simplification: the UI just shows label + raw value.)
    for (const t of watchTexts) {
      out.push({ label: `text:${JSON.stringify(t)}`, tokenId: -1, source: 'text' });
    }
    for (const raw of watchIds) {
      const tid = Number.parseInt(raw, 10);
      if (!Number.isFinite(tid) || seen.has(tid)) continue;
      seen.add(tid);
      const piece = tokenCache[tid];
      const suffix = piece ? ` ${JSON.stringify(piece)}` : '';
      out.push({ label: `id=${tid}${suffix}`, tokenId: tid, source: 'id' });
    }
    if (watchEos) {
      for (const tid of activeCaps?.eos_token_ids ?? []) {
        if (seen.has(tid)) continue;
        seen.add(tid);
        out.push({ label: `EOS:${tid}`, tokenId: tid, source: 'eos' });
      }
    }
    return out;
  });
  /**
   * Resolve one watch column at a given position by looking it up in
   * the row's ``watched`` array. For ``text`` columns we use the
   * positional index trick (text watches always come first in the
   * server's resolved order). For ``id`` / ``eos`` columns we know
   * the id and look it up directly.
   */
  function watchedAt(step: StepResult, col: WatchColumn, textIdxIfApplicable: number):
    TokenCandidate | null {
    if (col.source === 'text') {
      // Positional: text watches always lead the watched list in the
      // server's resolved order, so the i-th text column maps to the
      // i-th watched entry.
      const w = step.watched[textIdxIfApplicable];
      return w ? w.candidate : null;
    }
    return watchedById(step, col.tokenId);
  }

  let watchTextCount = $derived<number>(watchTexts.length);

  /**
   * Probability for the i-th manual pick, used to color the running-
   * completion view exactly like /generate's auto-generated tokens.
   * Forced tokens have ``null`` here (we don't know the model's
   * prob at the time the user typed them) and render plain grey.
   */
  function pickedProb(i: number): number | null {
    return pickedProbs[i] ?? null;
  }

  // ``promptSteps`` per the ``Backend.score_prompt`` contract is N rows
  // for an N-token prompt: the first N-1 carry the actual prompt token
  // as ``chosen``, and the trailing "(predict next)" row has
  // ``chosen=null`` and merely advertises what the model would emit
  // next. We render the running-completion prefix from the prompt rows
  // only -- the trailing prediction belongs in ``steps`` (the actual
  // generation), so including it here would double-count the first
  // generated token.
  let promptPrefixSteps = $derived(promptSteps.filter((s) => !!s?.chosen));
  // True iff ``promptSteps`` is an actual per-prompt-token scoring
  // (more than just a single-row chat-only fallback). For chat-only
  // providers ``promptSteps`` is a one-row next-token distribution --
  // still useful as a table but missing the per-prompt-token data we
  // need to colorize the prefix, so we leave the prompt rendering
  // plain grey in that case.
  let promptHasFullTokenization = $derived(promptPrefixSteps.length > 1);
  // First-token alignment: backends that go through the generic
  // ``Backend.score_prompt`` (dsbx-host-py, anything backed by a per-prefix
  // ``next_distribution`` loop) iterate ``range(1, N+1)`` and so the
  // FIRST prompt token is never emitted as a row's ``chosen`` -- it
  // only shows up as the ``context_text`` of row 1. Fireworks
  // ``stream_native_with_echo`` is the outlier: it iterates from
  // position 0 and emits ``chosen=tokens[i]`` for every echoed
  // position including the first, so the first token appears
  // naturally. To make the running-completion view consistent across
  // both shapes we detect the "first row starts at position > 0" case
  // and prepend the missing first-token text from row 1's
  // ``context_text``. Rendered as plain grey because we have no
  // model-assigned prob for position 0.
  let firstPromptTokenText = $derived.by<string | null>(() => {
    if (promptPrefixSteps.length === 0) return null;
    const first = promptPrefixSteps[0];
    if ((first.position ?? 0) <= 0) return null;
    return first.context_text ?? null;
  });

  // Inspect mode is implemented as ``max_tokens=1 + include_prompt=true``,
  // so the wire shape is: prompt_score (N rows + trailing predict-next
  // on backends that emit one) + ONE step. That single step is the
  // exact same "what does the model predict next" payload as the
  // trailing predict-next prompt-score row -- showing it as a
  // separate generation step both inflates the running-completion
  // view (adds a phantom token the user did NOT ask to generate) and
  // duplicates the predict-next table row. So in inspect mode we
  // hide ``steps`` rendering entirely. Generate / Manual modes are
  // unaffected.
  let hideStepsInRunningCompletion = $derived(lastMode === 'inspect');
  let hideGenerationStepsTable = $derived(lastMode === 'inspect');

  // ``eligible`` badge visibility: the badge marks candidates that
  // survived the current sampler's filter (``decision.kept``). Local
  // backends (dsbx-host-py, HF, llamacpp_py) run the sampler in-process
  // and populate ``kept`` on every step. Cloud-native streaming
  // (Fireworks ``stream_native`` / ``stream_native_with_echo``) runs
  // the sampler server-side and the upstream tells us nothing about
  // per-candidate eligibility -- so ``kept`` is left empty. Rendering
  // the badge based purely on per-row ``kept`` membership produces
  // "every greedy step shows ONE eligible / no Fireworks step shows
  // any" -- both technically correct but the asymmetry between
  // backends looks like a bug. We instead derive a page-level flag:
  // show the badge column only when ``kept`` is meaningful for AT
  // LEAST one rendered step, and surface a one-line note for the
  // server-side-sampler case so the user knows why the column is
  // absent rather than wondering whether nothing is eligible.
  let keptIsMeaningful = $derived<boolean>(
    steps.some((s) => (s.decision?.kept ?? []).length > 0)
  );
  let serverSideSamplerInUse = $derived<boolean>(
    steps.length > 0 && !keptIsMeaningful
  );
</script>

<Toast message={streamError} onClose={() => (streamError = null)} />

<!--
  ``biasable`` is the "click a token to add it to the logit_bias
  editor" snippet used by every prompt-logits / generation-steps
  table cell. Defined here once and rendered with
  ``{@render biasable(token_id, text, isSpecial, classes)}`` further
  down. When the backend doesn't support logit_bias OR the token id
  is synthetic (>= 1<<24, intern-only), we fall through to a plain
  ``TokenText`` so chat-only backends don't grow useless click
  affordances and biasing on a synthetic id (a no-op upstream) is
  prevented at the UI level.
-->
{#snippet biasable(
  tokenId: number | null | undefined,
  text: string,
  isSpecial: boolean,
  classes: string
)}
  <CompletionToken
    text={text}
    isSpecial={isSpecial}
    tokenId={tokenId ?? null}
    showMarkers={true}
    className={classes}
    probTitle={tokenId !== null && tokenId !== undefined && Number.isFinite(tokenId)
      ? `token id ${tokenId}`
      : ''}
    onWatch={addToWatchToken}
    onPrompt={addTokenToPrompt}
    onBias={logitBiasSupported ? (id) => addToBias(id) : undefined}
  />
{/snippet}

<div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
  <div class="card lg:col-span-1 space-y-3">
    <h2 class="text-lg font-semibold">Decode</h2>
    <p class="text-xs text-slate-400">
      Inspect, generate, or pick tokens manually. All three buttons hit a single endpoint
      (<span class="font-mono">/api/v1/generate/stream</span>) with different parameters; inspect is
      <span class="font-mono">max_tokens=1</span>, manual is per-pick <span class="font-mono">max_tokens=1 +
      prefix_token_ids = [your picks]</span>.
    </p>
    <BackendSelect bind:value={backend} onChange={onBackendChange} />
    {#if backendInfo?.model_reloadable}
      <!--
        Remote dsbx-serve hosts own a swappable model slot, so the Decode
        page lets you load/swap the model right here instead of bouncing to
        the Status page. ``onReady`` keeps the local ``model`` field in sync
        with whatever the host actually has loaded.
      -->
      <div>
        <span class="label">model (remote host)</span>
        <RemoteModelControl
          backend={backendInfo}
          compact={true}
          onReady={(m) => {
            if (m) model = m;
            maybeClearForSwitch();
          }}
        />
      </div>
    {:else}
      <ModelInput
        backend={backendInfo}
        bind:value={model}
        onChange={() => maybeClearForSwitch()}
      />
    {/if}
    <CapabilityBadges backend={backend} />
    <!--
      The prompt input itself lives in the RIGHT column now (the
      ``TokenComposer`` above "running completion"): it carries inline
      token-boundary highlighting plus the special-token palette, so the
      three artefacts a student compares sit side-by-side -- "what I typed
      / how it tokenizes" -> "what the model emits". This left card keeps
      the run controls (backend / model / sampler / stops / buttons).
    -->
    <div class="grid grid-cols-2 gap-2">
      <div>
        <label class="label" for="mt">max tokens</label>
        <input id="mt" type="number" min="1" max="200" class="input" bind:value={maxTokens} />
      </div>
      <div>
        <label class="label" for="alts">alternatives (top-k)</label>
        <input
          id="alts"
          type="number"
          min="1"
          max={altsMax}
          class="input"
          bind:value={alternatives}
        />
        <p class="text-[10px] text-slate-500 mt-0.5">
          max {altsMax} on {backendInfo?.label ?? 'this backend'}
        </p>
      </div>
      <div class="col-span-2">
        <label
          class="label flex items-center gap-2 {seedLocked ? 'text-slate-600' : ''}"
          for="seed"
          title={seedLocked
            ? 'Greedy sampling is deterministic (argmax) -- the seed has no effect.'
            : ''}
        >
          seed
          {#if seedLocked}
            <span class="text-[10px] uppercase tracking-wider text-slate-600 normal-case">
              no effect for greedy
            </span>
          {/if}
        </label>
        <input
          id="seed"
          type="number"
          class="input"
          bind:value={seed}
          disabled={seedLocked}
        />
      </div>
    </div>
    <div>
      <label class="label" for="sampler">Sampler</label>
      <select id="sampler" class="input font-mono" bind:value={sampler}>
        <option value="greedy">greedy</option>
        <option value="temperature">temperature</option>
        <option value="top_k">top_k</option>
        <option value="top_p">top_p</option>
        <option value="min_p">min_p</option>
        <option value="typical">typical</option>
        <option value="mirostat">mirostat (v2)</option>
      </select>
    </div>
    {#if sampler !== 'greedy'}
      <div class="grid grid-cols-2 gap-2">
        <div>
          <label class="label" for="T">T</label>
          <input id="T" type="number" step="0.05" min="0" max="5" class="input" bind:value={temperature} />
        </div>
        {#if sampler === 'top_k'}
          <div>
            <label class="label" for="stk">top_k</label>
            <input id="stk" type="number" min="1" max="200" class="input" bind:value={samplerTopK} />
          </div>
        {/if}
        {#if sampler === 'top_p'}
          <div>
            <label class="label" for="topp">top_p</label>
            <input id="topp" type="number" step="0.05" min="0" max="1" class="input" bind:value={topP} />
          </div>
        {/if}
        {#if sampler === 'min_p'}
          <div>
            <label class="label" for="minp">min_p</label>
            <input id="minp" type="number" step="0.05" min="0" max="1" class="input" bind:value={minP} />
          </div>
        {/if}
        {#if sampler === 'typical'}
          <div>
            <label class="label" for="typp">typical_p</label>
            <input id="typp" type="number" step="0.05" min="0" max="1" class="input" bind:value={typicalP} />
          </div>
        {/if}
        {#if sampler === 'mirostat'}
          <div>
            <label class="label" for="miro_t" title="Target surprise per step, in nats. Mirostat v2.">
              mirostat target (τ, nats)
            </label>
            <input id="miro_t" type="number" step="0.1" min="0.1" max="20" class="input" bind:value={mirostatTarget} />
          </div>
          <div>
            <label class="label" for="miro_lr">mirostat learning rate (η)</label>
            <input id="miro_lr" type="number" step="0.05" min="0" max="1" class="input" bind:value={mirostatLr} />
          </div>
        {/if}
        <!-- Penalties live alongside the sampler-specific knobs because
             they ride along on every sampler. Wire-side they only ship
             when their value differs from the no-op default. -->
        <div>
          <label class="label" for="repp" title="llama.cpp-style repetition penalty. 1.0 disables.">
            repetition penalty
          </label>
          <input id="repp" type="number" step="0.05" min="0.5" max="2" class="input" bind:value={repetitionPenalty} />
        </div>
        <div>
          <label class="label" for="freqp" title="OpenAI semantics: lp -= freq * count(token). 0 disables.">
            frequency penalty
          </label>
          <input id="freqp" type="number" step="0.1" min="-2" max="2" class="input" bind:value={frequencyPenalty} />
        </div>
        <div>
          <label class="label" for="presp" title="OpenAI semantics: lp -= pres if token already used. 0 disables.">
            presence penalty
          </label>
          <input id="presp" type="number" step="0.1" min="-2" max="2" class="input" bind:value={presencePenalty} />
        </div>
      </div>
    {/if}
    <ChipInput bind:values={stopTexts} label="Stop text" placeholder="e.g. '.'" />
    <ChipInput bind:values={stopIds} label="Stop ids" placeholder="token id" preserveSpace={false} />
    <ChipInput
      bind:values={watchTexts}
      label="Watch text"
      placeholder="e.g. ' Paris' (leading space preserved)"
      hint="Each chip is tokenized server-side; multi-token watches use their first id. Probabilities show in extra table columns."
    />
    <ChipInput
      bind:values={watchIds}
      label="Watch ids"
      placeholder="numeric token id"
      preserveSpace={false}
    />
    <label class="flex items-center gap-2 text-sm">
      <input type="checkbox" bind:checked={watchEos} class="accent-sky-500" />
      Watch EOS tokens
    </label>
    <!--
      The old "prepend tokens" chip-input + "fill BOS" button lived here.
      They're gone: the TokenComposer (right column) lets you insert the
      model's BOS -- or any special token -- directly at the start of the
      prompt, which is the same BOS-conditioning experiment expressed in
      the token stream itself. One composer, no separate splice field.
    -->
    {#if logitBiasSupported}
      <div class="space-y-1">
        <div class="flex items-center justify-between">
          <label class="label" title="Per-request additive bias on token logits. Range [-100, 100]; -100 effectively bans the token, +100 forces it (subject to grammar / stop overrides).">
            logit bias
            <span class="ml-1 normal-case text-[10px] text-slate-500">{logitBiasRows.length} rows</span>
          </label>
          <button
            type="button"
            class="text-xs px-2 py-0.5 rounded border border-slate-700 hover:border-slate-500"
            onclick={() => {
              logitBiasRows = [
                ...logitBiasRows,
                { id: crypto.randomUUID(), tokenId: '', bias: '' }
              ];
            }}
          >
            + add
          </button>
        </div>
        {#if logitBiasRows.length}
          <div class="space-y-1.5">
            {#each logitBiasRows as row (row.id)}
              {@const parsedId = Number.parseInt(row.tokenId, 10)}
              {@const cachedText =
                Number.isFinite(parsedId) ? tokenCache[parsedId] : undefined}
              {@const isSyntheticId =
                Number.isFinite(parsedId) && parsedId >= 1 << 24}
              <!--
                Grid layout (vs. flex) is the robust fix for "вёрстка
                поехала": fixed-width columns can't wrap onto a second
                line even when the parent panel is narrow, and the
                TokenText cell can truncate independently without
                affecting the bias / × columns. The numeric id is now
                deliberately small + secondary because the human-
                readable text is what the user actually reads when
                scanning the bias list (см. UX-feedback от 22 июня).
              -->
              <div class="grid grid-cols-[4rem_minmax(0,1fr)_4.5rem_1.5rem] items-center gap-1.5">
                <input
                  type="text"
                  inputmode="numeric"
                  list="token-id-suggestions"
                  autocomplete="off"
                  class="input w-full font-mono text-[11px] text-left px-1"
                  placeholder="id"
                  title={cachedText !== undefined
                    ? `id ${row.tokenId} = ${JSON.stringify(cachedText)}${isSyntheticId ? ' (synthetic intern id, not a real model token)' : ''}`
                    : 'pick from the autocomplete list or type a numeric id'}
                  bind:value={row.tokenId}
                />
                <div class="min-w-0 truncate text-xs">
                  {#if cachedText !== undefined}
                    <TokenText text={cachedText} className="font-mono text-xs text-slate-200" />
                    {#if isSyntheticId}
                      <span class="ml-1 text-[10px] text-amber-400" title="synthetic intern id from a chat-only backend — server won't honor a logit_bias on this id">⚠</span>
                    {/if}
                  {:else if row.tokenId !== ''}
                    <span class="text-[10px] text-slate-500" title="no text known for this id yet; the autocomplete list shows tokens we've seen in the prompt or generation">unknown token</span>
                  {:else}
                    <span class="text-[10px] text-slate-600">— text appears here</span>
                  {/if}
                </div>
                <input
                  type="number"
                  step="0.5"
                  min="-100"
                  max="100"
                  class="input w-full font-mono text-[11px] text-right px-1"
                  placeholder="bias"
                  bind:value={row.bias}
                />
                <button
                  type="button"
                  class="text-xs leading-none px-1.5 py-1 rounded border border-slate-700 hover:border-rose-500 hover:text-rose-300"
                  title="remove this entry"
                  aria-label="remove bias row"
                  onclick={() => {
                    logitBiasRows = logitBiasRows.filter((r) => r.id !== row.id);
                  }}
                >
                  ×
                </button>
              </div>
            {/each}
          </div>
        {:else}
          <div class="text-xs text-slate-500">
            no entries — click any token in the
            <em>prompt logits</em> / <em>generation steps</em> tables
            to add it (default bias <span class="font-mono">−100</span>,
            i.e. ban), or hit <em>add</em> and type a numeric id
            (autocomplete shows tokens seen so far).
          </div>
        {/if}
        {#if tokenCacheEntries.length}
          <datalist id="token-id-suggestions">
            {#each tokenCacheEntries as e (e.id)}
              <option value={String(e.id)} label={`${e.id} · ${e.text}`}></option>
            {/each}
          </datalist>
        {/if}
      </div>
    {/if}
    <div class="space-y-2">
      <label
        class="flex items-center gap-2 text-sm {respectEosLocked
          ? 'text-slate-500 cursor-not-allowed'
          : ''}"
        title={respectEosLocked
          ? 'This cloud provider has no ignore_eos field; the upstream always halts on EOS.'
          : 'Uncheck to keep generating past EOS (Fireworks: ships ignore_eos:true on the wire).'}
      >
        <input
          type="checkbox"
          bind:checked={respectEos}
          class="accent-sky-500"
          disabled={respectEosLocked}
        />
        respect EOS
        {#if respectEosLocked}
          <span class="text-[10px] uppercase tracking-wider text-slate-600">no ignore_eos on this backend</span>
        {/if}
      </label>
      <label
        class="flex items-center gap-2 text-sm"
        title={promptScoreDegraded
          ? 'This backend cannot score whole prompts; you will get a single next-token row instead.'
          : ''}
      >
        <input type="checkbox" bind:checked={includePrompt} class="accent-sky-500" />
        include prompt logits
        {#if promptScoreDegraded && includePrompt}
          <span class="text-[10px] uppercase tracking-wider text-amber-500/80 normal-case">
            next-token only (chat-only backend)
          </span>
        {/if}
        {#if includePrompt && combinedEchoStreamSupported}
          <span
            class="text-[10px] uppercase tracking-wider text-emerald-500/80 normal-case"
            title="This backend runs include_prompt in ONE request (echo+stream) instead of two."
          >
            1-request mode
          </span>
        {/if}
      </label>
      {#if includePrompt && combinedEchoStreamSupported}
        <div class="ml-6 flex items-center gap-2 text-xs text-slate-400">
          <label for="echo-last" class="cursor-help" title="0 = echo every prompt token. >0 = echo logprobs only for the last N prompt tokens (Fireworks echo_last).">
            echo last N
          </label>
          <input
            id="echo-last"
            type="number"
            min="0"
            class="input w-24 font-mono text-xs"
            bind:value={echoLast}
          />
          <span class="text-slate-600">
            (0 = whole prompt)
          </span>
        </div>
      {/if}
      <label class="flex items-center gap-2 text-sm">
        <input type="checkbox" bind:checked={showMarkers} class="accent-sky-500" />
        show whitespace markers (<span class="font-mono">␣ ↵ →</span>) in completion
      </label>
    </div>
    {#if serviceTierSupported}
      <div>
        <label class="label" for="tier">
          service tier
          <span class="text-[10px] uppercase tracking-wider text-slate-600 normal-case">
            provider-specific
          </span>
        </label>
        <select id="tier" class="input font-mono" bind:value={serviceTier}>
          <option value="default">default (shared serverless pool)</option>
          <option value="priority">priority (dedicated lane)</option>
        </select>
        <p class="mt-1 text-[11px] text-slate-500 leading-snug">
          On Fireworks, <span class="font-mono">priority</span> routes
          the request to a dedicated, lower-latency inference pool
          (higher per-token cost, no rate-limit sharing).
          <span class="font-mono">default</span> uses the shared
          serverless pool — cheaper, but subject to occasional cold
          starts and rate caps. Pick <span class="font-mono">priority</span>
          for latency-sensitive experiments (ttft matters) or when the
          default pool returns 429s under load.
        </p>
      </div>
    {/if}
  </div>

  <div class="lg:col-span-2 space-y-3">
    <div class="card py-2 px-3">
      <!--
        The prompt composer: an editable field with INLINE token-boundary
        highlighting plus a model-specific special-token palette. It lives
        here (right column, above "running completion") so the student sees
        "what I typed / how it tokenizes" right next to "what the model
        emits". ``enabled`` gates the highlight + palette on a real local
        tokenizer; without one it degrades to a plain textarea.

        The run controls (inspect / generate / manual) sit DIRECTLY under
        the composer -- they used to live at the bottom of the left config
        card where they got lost far from the input the user is editing.
      -->
      <TokenComposer
        bind:value={prompt}
        backend={backend}
        model={model}
        enabled={localTokenizeSupported}
      />
      <div class="mt-3 grid grid-cols-3 gap-2">
        <button
          class="btn flex-1 {lastMode === 'inspect' && !busy ? 'btn-primary' : 'btn-ghost'}"
          onclick={runInspect}
          disabled={busy || !backend || generationDisabled || promptEmpty}
          title={generationDisabled
            ? generationDisabledNote
            : promptEmpty
              ? emptyPromptNote
              : 'Score every prompt position (max_tokens=1 + include_prompt). Same wire path as generate; just stops after one emitted token.'}
        >
          {busy && lastMode === 'inspect' ? '…' : 'inspect'}
        </button>
        <button
          class="btn flex-1 {lastMode === 'generate' && !busy ? 'btn-primary' : 'btn-ghost'}"
          onclick={runGenerate}
          disabled={busy || !backend || generationDisabled || promptEmpty}
          title={generationDisabled
            ? generationDisabledNote
            : promptEmpty
              ? emptyPromptNote
              : 'Stream N tokens with the current sampler.'}
        >
          {busy && lastMode === 'generate' ? '…' : 'generate'}
        </button>
        <button
          class="btn flex-1 {manualMode ? 'btn-primary' : 'btn-ghost'}"
          onclick={enterManual}
          disabled={busy || !backend || generationDisabled || promptEmpty}
          title={generationDisabled
            ? generationDisabledNote
            : promptEmpty
              ? emptyPromptNote
              : 'Open the inline picker: pick or force each token by hand. Browser state only; one /generate/stream call per pick (Fireworks reuses KV cache via session_id + prompt_cache_key).'}
        >
          {busy && lastMode === 'manual' ? '…' : 'manual'}
        </button>
      </div>
      {#if busy}
        <div class="mt-2 flex justify-end">
          <button class="btn btn-ghost text-xs" onclick={cancel}>stop streaming</button>
        </div>
      {/if}
      {#if promptEmpty && !generationDisabled}
        <p class="mt-2 text-[11px] text-amber-400 leading-snug">
          {emptyPromptNote}
        </p>
      {/if}
      {#if generationDisabled}
        <p class="mt-1 text-[11px] text-amber-400 leading-snug">
          {generationDisabledNote || 'generation disabled for this backend'}
        </p>
      {/if}
      {#if manualMode}
        <p class="mt-2 text-[11px] text-sky-400/80 leading-snug">
          manual mode active — pick a candidate in the right panel; each pick fires
          one <span class="font-mono">/generate/stream</span> call with
          <span class="font-mono">prefix_token_ids = [your picks]</span>.
          <button
            type="button"
            class="ml-1 underline decoration-dotted hover:text-sky-200"
            onclick={exitManual}
          >exit manual</button>
        </p>
      {/if}
    </div>

    <div class="card">
      <div class="flex items-center justify-between mb-1">
        <div class="flex items-center gap-3">
          <span class="text-xs uppercase tracking-wider text-slate-500">running completion</span>
          {#if completionText.length > 0}
            <button
              type="button"
              class="btn btn-ghost text-[11px] py-0.5 px-2"
              onclick={moveCompletionToPrompt}
              disabled={busy}
              title="Fold the run's prompt + the whole generated continuation into the prompt box, so you can keep generating from exactly what's shown here."
            >→ move to prompt</button>
          {/if}
        </div>
        <div class="text-[10px] uppercase tracking-wider text-slate-600 flex items-center gap-2">
          <span class="inline-block w-3 h-3 rounded bg-emerald-500/40"></span>≥80%
          <span class="inline-block w-3 h-3 rounded bg-sky-500/40"></span>≥50%
          <span class="inline-block w-3 h-3 rounded bg-amber-500/40"></span>≥25%
          <span class="inline-block w-3 h-3 rounded bg-orange-500/40"></span>≥10%
          <span class="inline-block w-3 h-3 rounded bg-rose-500/40"></span>&lt;10%
        </div>
      </div>
      <div class="font-mono text-sm text-slate-200 whitespace-pre-wrap min-h-[2.5rem] leading-relaxed">
        {#if promptHasFullTokenization}{#if firstPromptTokenText !== null}<TokenInline
              text={firstPromptTokenText}
              showMarkers={showMarkers}
              bgClass=""
              title="First prompt token · INPUT only, not predicted. Autoregressive models compute P(next | prior tokens); position 0 has no prior, so the model has nothing to predict from (unless you prepend a BOS marker — currently not done on this backend)."
            />{/if}{#each promptPrefixSteps as ps}{@const lp = ps.chosen?.logprob ?? null}{@const p = probFromLogprob(lp)}{@const isUnscoredFirst = (!ps.candidates || ps.candidates.length === 0) && (lp === null || !Number.isFinite(lp))}<TokenInline
              text={ps.chosen?.text ?? ''}
              isSpecial={ps.chosen?.is_special ?? false}
              showMarkers={showMarkers}
              bgClass={isUnscoredFirst ? '' : tokenBackgroundClass(p)}
              title={isUnscoredFirst
                ? 'First prompt token · INPUT only, not predicted. The upstream returned no logprob for position 0 because autoregressive models have nothing to predict from before the first token.'
                : `prompt · p=${p !== null ? ((p ?? 0) * 100).toFixed(2) + '%' : '?'}`}
            />{/each}{:else}<span class="text-slate-400">{runningPromptDisplay}</span>{/if}{#each pickedTexts as pt, i}{@const pp = pickedProb(i)}<TokenInline
            text={pt}
            showMarkers={showMarkers}
            bgClass={tokenBackgroundClass(pp)}
            title={`manual pick #${i + 1}${pp !== null ? ` · p=${(pp * 100).toFixed(2)}%` : ' · forced'}`}
          />{/each}{#if !hideStepsInRunningCompletion}{#each steps as s}<CompletionToken
            text={s.decision.token_text}
            tokenId={s.decision.token_id}
            showMarkers={showMarkers}
            bgClass={tokenBackgroundClass(chosenProb(s))}
            candidates={s.step_result.candidates}
            probTitle={`p=${chosenProb(s) !== null ? ((chosenProb(s) ?? 0) * 100).toFixed(2) + '%' : '?'}`}
            onWatch={addToWatchToken}
            onPrompt={addTokenToPrompt}
            onFind={() => findStepInList(s.step)}
            onBias={logitBiasSupported ? (id) => addToBias(id) : undefined}
          />{/each}{/if}{#if busy}<span class="animate-pulse text-sky-400">▌</span>{/if}
      </div>
      {#if stopReason}
        <div class="text-xs text-slate-500 mt-2">stopped: <span class="font-mono">{stopReason}</span></div>
      {/if}
      {#if usage}
        <div class="text-xs text-slate-500 mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
          <span>
            <span class="font-mono tabular-nums {usage.requests > 1 ? 'text-amber-400' : 'text-slate-300'}">{usage.requests}</span>
            request{usage.requests === 1 ? '' : 's'}
          </span>
          <span>
            in <span class="font-mono tabular-nums text-slate-300">{usage.prompt_tokens ?? '—'}</span>
            ·
            out <span class="font-mono tabular-nums text-slate-300">{usage.completion_tokens ?? '—'}</span>
            tokens
            {#if usage.total_tokens !== null}
              <span class="text-slate-600">(total <span class="font-mono tabular-nums">{usage.total_tokens}</span>)</span>
            {/if}
          </span>
          {#each usage.notes as note}
            <span class="text-amber-400 normal-case">⚠ {note}</span>
          {/each}
        </div>
      {/if}
    </div>

    {#if manualMode}
      <div class="card space-y-3 border-sky-700/40">
        <div class="flex items-center justify-between">
          <div class="text-xs uppercase tracking-wider text-sky-300">
            manual picker
            <span class="ml-2 normal-case text-[10px] text-slate-500">
              {pickedIds.length} pick{pickedIds.length === 1 ? '' : 's'}
              · session <span class="font-mono">{manualSessionId.slice(0, 8)}</span>
            </span>
          </div>
          <button
            type="button"
            class="text-xs underline decoration-dotted text-slate-400 hover:text-slate-200"
            onclick={exitManual}
          >exit manual</button>
        </div>

        {#if pickedIds.length > 0}
          <div class="flex flex-wrap items-center gap-1 text-xs">
            <span class="text-slate-500 uppercase tracking-wider text-[10px] mr-1">picks:</span>
            {#each pickedIds as tid, i}
              <button
                type="button"
                class="px-1.5 py-0.5 rounded border border-slate-700 hover:border-rose-500 hover:text-rose-300 font-mono text-[11px]"
                title={`click to undo back to step ${i} (drops picks ${i + 1}..${pickedIds.length})`}
                onclick={() => manualUndoTo(i)}
              >
                <TokenText text={pickedTexts[i] || `id=${tid}`} className="font-mono text-[11px]" />
                <span class="text-slate-600">·{tid}</span>
              </button>
            {/each}
            <button
              type="button"
              class="ml-1 px-2 py-0.5 rounded border border-slate-700 hover:border-amber-500 hover:text-amber-300 text-[11px]"
              onclick={manualUndo}
              disabled={busy || pickedIds.length === 0}
              title="undo last pick"
            >undo</button>
          </div>
        {/if}

        <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
          <div>
            <label class="label" for="force-text">force text</label>
            <div class="flex gap-1">
              <input
                id="force-text"
                type="text"
                class="input flex-1 font-mono"
                bind:value={forceText}
                placeholder="e.g. ' the'"
                onkeydown={(e) => { if (e.key === 'Enter') pickForceText(); }}
              />
              <button
                class="btn btn-ghost"
                onclick={pickForceText}
                disabled={busy || !forceText}
              >+</button>
            </div>
            <p class="text-[10px] text-slate-500 mt-0.5">
              Tokenized server-side; multi-token strings expand into N picks.
            </p>
          </div>
          <div>
            <label class="label" for="force-id">force id</label>
            <div class="flex gap-1">
              <input
                id="force-id"
                type="text"
                inputmode="numeric"
                class="input flex-1 font-mono"
                bind:value={forceId}
                list="token-id-suggestions"
                placeholder="numeric id"
                onkeydown={(e) => { if (e.key === 'Enter') pickForceId(); }}
              />
              <button
                class="btn btn-ghost"
                onclick={pickForceId}
                disabled={busy || !forceId}
              >+</button>
            </div>
          </div>
        </div>

        {#if manualDistribution}
          <div>
            <div class="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
              next-token distribution (click a row to pick)
            </div>
            <table class="w-full text-sm">
              <thead class="text-xs text-slate-400 border-b border-slate-800">
                <tr>
                  <th class="table-cell text-left">rank</th>
                  <th class="table-cell text-left">token</th>
                  <th class="table-cell text-left">prob</th>
                  <th class="table-cell text-left">id</th>
                  <th class="table-cell text-left">sampler</th>
                </tr>
              </thead>
              <tbody>
                {#each manualDistribution.candidates.slice(0, alternatives) as c}
                  {@const p = c.logprob !== null ? Math.exp(c.logprob) : null}
                  <tr
                    class="border-b border-slate-800/60 hover:bg-slate-800/40 cursor-pointer"
                    onclick={() => pickCandidate(c)}
                    data-token-row
                  >
                    <td class="table-cell font-mono text-slate-400">{c.rank + 1}</td>
                    <td class="table-cell">
                      <TokenText text={c.text} isSpecial={c.is_special} className="font-mono" />
                    </td>
                    <td class="table-cell w-32"><ConfidenceBar prob={p} /></td>
                    <td class="table-cell font-mono text-xs text-slate-500 tabular-nums">{c.token_id}</td>
                    <td class="table-cell text-[10px] uppercase tracking-wider text-slate-500">
                      {c.sampling_mask_count !== undefined && c.sampling_mask_count !== null
                        ? `mask=${c.sampling_mask_count}`
                        : ''}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}
      </div>
    {/if}

    {#if rawOutput && rawOutputSupported}
      <div class="card">
        <div class="text-xs uppercase tracking-wider text-slate-500 mb-2">
          what the model saw
          <span class="ml-2 normal-case text-[10px] text-slate-600">
            from the provider's raw_output block -- shows the prompt AFTER chat-template
            rendering / BOS injection, and the grammar (if any) the server compiled.
          </span>
        </div>
        {#if Array.isArray(rawOutput.prompt_fragments) && rawOutput.prompt_fragments.length}
          <div class="mb-3">
            <div class="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
              prompt fragments
            </div>
            <div class="font-mono text-xs leading-relaxed bg-slate-900/40 p-2 rounded border border-slate-800/60 break-all">
              {#each rawOutput.prompt_fragments as frag, i}
                <span class="text-slate-300" title={`fragment ${i}`}>{frag}</span>
                {#if i < (rawOutput.prompt_fragments?.length ?? 0) - 1}
                  <span class="text-slate-700">·</span>
                {/if}
              {/each}
            </div>
          </div>
        {/if}
        {#if Array.isArray(rawOutput.prompt_token_ids) && rawOutput.prompt_token_ids.length}
          <div class="mb-3">
            <div class="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
              prompt token ids ({rawOutput.prompt_token_ids.length} tokens)
            </div>
            <div class="font-mono text-xs text-slate-400 break-all max-h-32 overflow-y-auto">
              {rawOutput.prompt_token_ids.join(', ')}
            </div>
          </div>
        {/if}
        {#if rawOutput.grammar}
          <div class="mb-3">
            <div class="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
              compiled grammar
            </div>
            <pre class="font-mono text-xs text-slate-300 bg-slate-900/40 p-2 rounded border border-slate-800/60 overflow-x-auto">{JSON.stringify(rawOutput.grammar, null, 2)}</pre>
          </div>
        {/if}
        {#if Object.keys(rawOutput).some((k) => !['prompt_fragments', 'prompt_token_ids', 'grammar'].includes(k))}
          <details class="text-xs">
            <summary class="cursor-pointer text-slate-500 hover:text-slate-300">
              other fields ({Object.keys(rawOutput).filter((k) => !['prompt_fragments', 'prompt_token_ids', 'grammar'].includes(k)).length})
            </summary>
            <pre class="mt-2 font-mono text-xs text-slate-400 bg-slate-900/40 p-2 rounded border border-slate-800/60 overflow-x-auto">{JSON.stringify(
                Object.fromEntries(
                  Object.entries(rawOutput).filter(
                    ([k]) => !['prompt_fragments', 'prompt_token_ids', 'grammar'].includes(k)
                  )
                ),
                null,
                2
              )}</pre>
          </details>
        {/if}
      </div>
    {/if}

    {#if perf && perfPanelSupported}
      <div class="card">
        <div class="text-xs uppercase tracking-wider text-slate-500 mb-2">
          server timings
          <span class="ml-2 normal-case text-[10px] text-slate-600">
            reported by the provider's perf_metrics block
          </span>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-1 text-sm">
          {#each Object.entries(perf) as [key, value]}
            <div class="flex flex-col">
              <span class="text-[10px] uppercase tracking-wider text-slate-500">{key}</span>
              <span class="font-mono text-slate-200 truncate" title={typeof value === 'object' ? JSON.stringify(value) : String(value)}>
                {#if typeof value === 'number'}
                  {#if key.includes('duration') || key.includes('time') || key.includes('latency')}
                    {(value * 1000).toFixed(1)} ms
                  {:else}
                    {value}
                  {/if}
                {:else if value === null || value === undefined}
                  —
                {:else if typeof value === 'object'}
                  {JSON.stringify(value)}
                {:else}
                  {String(value)}
                {/if}
              </span>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    {#if promptSteps.length}
      <div class="card overflow-x-auto">
        <div class="text-xs uppercase tracking-wider text-slate-500 mb-2">
          prompt logits
          {#if promptNote}
            <span class="ml-2 normal-case text-amber-400">{promptNote}</span>
          {/if}
        </div>
        <table class="w-full text-sm">
          <thead class="text-xs text-slate-400 border-b border-slate-800">
            <tr>
              <th class="table-cell text-left">pos</th>
              <th class="table-cell text-left">previous</th>
              <th class="table-cell text-left">actual next</th>
              <th class="table-cell text-left">prob</th>
              {#if samplingMaskSupported}
                <th
                  class="table-cell text-left"
                  title="Number of tokens above the active sampling threshold (Fireworks sampling_mask=count). Lower means a more constrained next-token distribution. Per-row, not per-candidate."
                  >mask</th
                >
              {/if}
              <th
                class="table-cell text-left"
                title="Top-K candidates by logprob, returned by the backend. For prompt-logits these are the model's predictions BEFORE sampling at each position."
              >
                top alts (rank 1..{alternatives})
              </th>
              {#each watchColumns as w}
                <th class="table-cell text-left" title={`watch column · ${w.label}`}>{w.label}</th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each promptSteps as s}
              {@const chosenLP = s.chosen?.logprob ?? null}
              {@const chosenP = probFromLogprob(chosenLP)}
              <!--
                An "unscored" prompt position is one where the upstream
                returned NO ``top_logprobs`` (Fireworks emits this for
                position 0 of every echoed prompt: the model has no
                prior context to predict a distribution from, so
                ``candidates`` is empty AND ``chosen.logprob`` is a
                placeholder we already coerced to NaN backend-side).
                We surface this as a pedagogical row so the user
                understands the autoregressive nature of the model
                instead of seeing an empty alts column and an
                ambiguous "?" prob with no explanation. Position 0 is
                the canonical case; the same rule applies to any other
                position that comes back empty (defensive).
              -->
              {@const isUnscored =
                (!s.candidates || s.candidates.length === 0) &&
                (chosenLP === null || !Number.isFinite(chosenLP))}
              <!--
                When the user is using the "prepend tokens" workflow,
                rows at positions 1..prependCount-1 carry CHOSEN =
                another prepended id (the model predicting its own
                injected seed sequence -- mostly noise, but still
                meaningful: see what the model thinks "after BOS"
                etc.). The row at position == prependCount is the
                interesting one: its chosen is the user's FIRST
                actual prompt token, scored against the
                BOS/prepended-conditioned distribution. We highlight
                that row with an emerald left-border + a discrete
                "BOS-conditioned" badge so the pedagogical payoff is
                obvious at a glance instead of buried in the
                "previous" column.
              -->
              {@const promptScorePrependCount = lastRunPrependCount}
              {@const isBosConditioned =
                promptScorePrependCount > 0 && s.position === promptScorePrependCount}
              {@const isPrependedSeedRow =
                promptScorePrependCount > 0 && s.position < promptScorePrependCount}
              <tr
                class="border-b border-slate-800/60 {isBosConditioned
                  ? 'border-l-2 border-l-emerald-500/60'
                  : isPrependedSeedRow
                    ? 'border-l-2 border-l-slate-700/60'
                    : ''}"
              >
                <td class="table-cell font-mono text-slate-400"
                  >{s.position}{#if isPrependedSeedRow}<span
                      class="ml-1 text-[9px] uppercase tracking-wider text-slate-600"
                      title="Row from prepended-seed context: chosen is another prepended token (positions 1..K-1, where K=number of prepended ids). The pedagogically interesting row is the one marked 'BOS-conditioned' just below — its chosen is your first real prompt token."
                      >seed</span
                    >{:else if isBosConditioned}<span
                      class="ml-1 text-[9px] uppercase tracking-wider text-emerald-400/90"
                      title="BOS-conditioned: chosen is your first REAL prompt token, finally scored against a distribution the model actually computed (the rows above were the prepended seed; this row's chosen is the first thing you actually typed)."
                      >BOS-conditioned</span
                    >{/if}</td
                >
                <td class="table-cell font-mono text-slate-500 text-xs">
                  {#if isUnscored}
                    <span
                      class="text-[10px] uppercase tracking-wider text-slate-600"
                      title="No prior context to condition on. Autoregressive models compute P(next | previous), so position 0 has nothing to predict from unless you prepend a BOS / chat-template marker."
                    >input only</span>
                  {:else if s.context_text !== null && s.context_text !== undefined}
                    <TokenText text={s.context_text} className="font-mono text-xs" />
                  {/if}
                </td>
                <td class="table-cell">
                  {#if s.chosen}
                    {@render biasable(s.chosen.token_id, s.chosen.text, s.chosen.is_special, 'font-mono')}
                  {:else}
                    <span class="text-slate-500">?</span>
                  {/if}
                </td>
                <td class="table-cell w-40">
                  {#if isUnscored}
                    <span
                      class="text-[10px] text-slate-600 italic"
                      title="The model didn't score this position; nothing to plot."
                    >—</span>
                  {:else}
                    <ConfidenceBar prob={chosenP} />
                  {/if}
                </td>
                {#if samplingMaskSupported}
                  <td class="table-cell font-mono text-xs text-slate-400 tabular-nums">
                    {s.candidates[0]?.sampling_mask_count ?? '?'}
                  </td>
                {/if}
                <td class="table-cell">
                  {#if isUnscored}
                    <span
                      class="text-[11px] text-slate-500 italic leading-snug"
                      title="No model prediction is available at this position. Autoregressive language models predict P(next | prior tokens); position 0 has no prior, so there is nothing to predict (unless you give the model a BOS / begin-of-sequence marker, which would add an extra forward pass). The first prompt token is shown only as INPUT to the model — it is what the model conditions on, not what it predicts."
                    >no model prediction · autoregressive model needs prior context (BOS)</span>
                  {:else}
                    <div class="flex flex-col gap-0.5">
                      {#each s.candidates.slice(0, alternatives) as c}
                        <div class="flex items-center gap-2">
                          {@render biasable(c.token_id, c.text, c.is_special, 'font-mono text-xs')}
                          <span
                            class="text-xs text-slate-500 tabular-nums"
                            title={c.logprob !== null && Number.isFinite(c.logprob)
                              ? `logprob=${c.logprob.toFixed(3)} · prob=${Math.exp(c.logprob).toExponential(2)}`
                              : 'no logprob reported by upstream'}
                          >
                            {formatProbPct(c.logprob)}
                          </span>
                        </div>
                      {/each}
                    </div>
                  {/if}
                </td>
                {#each watchColumns as w, wi}
                  {@const cand = watchedAt(s, w, w.source === 'text' ? wi : 0)}
                  <td class="table-cell">
                    {#if cand}
                      <ConfidenceBar prob={cand.logprob !== null ? Math.exp(cand.logprob) : null} />
                    {:else}
                      <span class="text-slate-600 text-xs">—</span>
                    {/if}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    {#if steps.length && !hideGenerationStepsTable}
      <div class="card overflow-x-auto">
        <div class="text-xs uppercase tracking-wider text-slate-500 mb-2 flex items-center flex-wrap gap-x-3">
          <span>generation steps</span>
          {#if serverSideSamplerInUse}
            <span
              class="normal-case text-[10px] text-slate-500"
              title="The provider's server-side sampler ran (Fireworks native streaming). Per-candidate eligibility (which tokens survived the sampler filter) is not reported by the upstream, so the 'eligible' marker is hidden for this run. Switch to a custom sampler or a full-vocab backend (dsbx-host-py / HF / llamacpp-py) to see it."
            >
              sampler ran server-side · per-candidate eligibility not reported
            </span>
          {/if}
        </div>
        <table class="w-full text-sm">
          <thead class="text-xs text-slate-400 border-b border-slate-800">
            <tr>
              <th class="table-cell text-left">step</th>
              <th class="table-cell text-left">chosen</th>
              <th class="table-cell text-left">prob</th>
              {#if samplingMaskSupported}
                <th
                  class="table-cell text-left"
                  title="Number of tokens that survived server-side sampling filters (sampling_mask=count) at this step."
                  >mask</th
                >
              {/if}
              <th
                class="table-cell text-left"
                title="Top-K candidates by logprob, returned by the backend BEFORE the sampler runs. With a stochastic sampler (e.g. temperature=1) the chosen token may have been sampled from outside this top-K — when that happens we still surface it as a row marked 'chosen, outside top-K'."
              >
                top alts (rank 1..{alternatives})
              </th>
              {#each watchColumns as w}
                <th class="table-cell text-left" title={`watch column · ${w.label}`}>{w.label}</th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each steps as s}
              {@const chosenInTopK = s.step_result.candidates
                .slice(0, alternatives)
                .some((c) => c.token_id === s.decision.token_id)}
              {@const chosenCand = s.step_result.chosen}
              <tr id={`gen-step-${s.step}`} class="border-b border-slate-800/60" data-token-row>
                <td class="table-cell font-mono text-slate-400">{s.step}</td>
                <td class="table-cell">
                  {@render biasable(
                    s.decision.token_id,
                    s.decision.token_text,
                    false,
                    'font-mono'
                  )}
                </td>
                <td class="table-cell w-40">
                  <ConfidenceBar prob={chosenProb(s)} />
                </td>
                {#if samplingMaskSupported}
                  <td class="table-cell font-mono text-xs text-slate-400 tabular-nums">
                    {s.step_result.candidates[0]?.sampling_mask_count ?? '?'}
                  </td>
                {/if}
                <td class="table-cell">
                  <div class="flex flex-col gap-0.5">
                    {#if !chosenInTopK && chosenCand}
                      <!--
                        The sampler picked something the backend didn't
                        return in top-K (Fireworks caps top_logprobs at
                        5; with temperature=1 the chosen token can sit
                        at rank 6+ in the real distribution). Render
                        the chosen as a synthetic top row with its
                        real logprob (the provider sends the sampled
                        token's logprob on the choice itself even when
                        omitting it from top_logprobs), so the user
                        sees WHAT was picked alongside the visible
                        alternatives.

                        Display rules (avoiding the historical "0.0%"
                        lie that conflated three different things):
                          * NaN / null logprob   -> "?"  (no data at all)
                          * 0 < prob < 0.1%      -> "<0.1%" (real but tiny;
                            "0.0%" rounded down made tokens look impossible
                            even though the sampler clearly picked them)
                          * prob >= 0.1%         -> "X.X%"  (normal display)
                      -->
                      {@const chosenLp = chosenCand.logprob}
                      <div class="flex items-center gap-2 border-l-2 border-sky-500/60 pl-1.5">
                        {@render biasable(
                          chosenCand.token_id,
                          chosenCand.text,
                          chosenCand.is_special,
                          'font-mono text-xs'
                        )}
                        <span
                          class="text-xs text-slate-500 tabular-nums"
                          title={chosenLp !== null && Number.isFinite(chosenLp)
                            ? `logprob=${chosenLp.toFixed(3)} · prob=${Math.exp(chosenLp).toExponential(2)} · the provider DOES report a real logprob for the sampled token even when it omits it from top_logprobs; the value is just very small, which is exactly why this token is outside top-K`
                            : 'provider did not report a logprob for this token (chosen is outside top-K AND logprob is missing)'}
                        >
                          {formatProbPct(chosenLp)}
                        </span>
                        <span
                          class="text-[9px] uppercase tracking-wider text-sky-300/90"
                          title="The sampler picked this token; it falls outside the top-{alternatives} the provider returned (top_logprobs cap)."
                        >← chosen · outside top-{alternatives}</span>
                      </div>
                    {/if}
                    {#each s.step_result.candidates.slice(0, alternatives) as c}
                      {@const eligible = (s.decision.kept ?? []).some((k) => k.token_id === c.token_id)}
                      {@const isChosen = c.token_id === s.decision.token_id}
                      <div class="flex items-center gap-2">
                        {@render biasable(c.token_id, c.text, c.is_special, 'font-mono text-xs')}
                        <span
                          class="text-xs text-slate-500 tabular-nums"
                          title={c.logprob !== null && Number.isFinite(c.logprob)
                            ? `logprob=${c.logprob.toFixed(3)} · prob=${Math.exp(c.logprob).toExponential(2)}`
                            : 'no logprob reported by upstream'}
                        >
                          {formatProbPct(c.logprob)}
                        </span>
                        {#if isChosen}
                          <span
                            class="text-[9px] uppercase tracking-wider text-sky-300/90"
                            title="The sampler picked this candidate."
                          >← chosen</span>
                        {/if}
                        {#if eligible && keptIsMeaningful}
                          <span
                            class="text-[9px] uppercase tracking-wider text-emerald-400/80"
                            title="Survived the current sampler's filter (decision.kept). For greedy that's just the argmax; for top_p / top_k / min_p / typical / mirostat it's the post-filter candidate set the sampler picked FROM (so any of these could have been chosen with the right random draw)."
                          >eligible</span>
                        {/if}
                      </div>
                    {/each}
                  </div>
                </td>
                {#each watchColumns as w, wi}
                  {@const cand = watchedAt(s.step_result, w, w.source === 'text' ? wi : 0)}
                  <td class="table-cell">
                    {#if cand}
                      <ConfidenceBar prob={cand.logprob !== null ? Math.exp(cand.logprob) : null} />
                    {:else}
                      <span class="text-slate-600 text-xs">—</span>
                    {/if}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>
</div>

<style>
  /* Transient highlight applied to a generation-steps row when the user
     picks "find in list" from a running-completion token, so the eye
     lands on the exact step they clicked. */
  :global(tr.row-flash) {
    animation: row-flash 1.6s ease-out;
  }
  @keyframes row-flash {
    0% {
      background: rgb(56 189 248 / 0.35);
    }
    100% {
      background: transparent;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    :global(tr.row-flash) {
      animation: none;
      background: rgb(56 189 248 / 0.18);
    }
  }
</style>
