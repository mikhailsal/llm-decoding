<script lang="ts">
  import { onMount } from 'svelte';
  import BackendSelect from '$lib/components/BackendSelect.svelte';
  import ModelInput from '$lib/components/ModelInput.svelte';
  import CapabilityBadges from '$lib/components/CapabilityBadges.svelte';
  import ChipInput from '$lib/components/ChipInput.svelte';
  import ConfidenceBar from '$lib/components/ConfidenceBar.svelte';
  import TokenText from '$lib/components/TokenText.svelte';
  import TokenInline from '$lib/components/TokenInline.svelte';
  import Toast from '$lib/components/Toast.svelte';
  import { apiStream } from '$lib/api';
  import { info } from '$lib/stores/info';
  import { probFromLogprob, tokenBackgroundClass } from '$lib/render';
  import type {
    GenStep,
    StepResult,
    TokenCandidate,
    BackendInfo,
    UsagePayload,
    PerfMetricsPayload,
    RawOutputPayload
  } from '$lib/types';

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
  let includePrompt = $state(false);
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

  let backendInfo = $derived<BackendInfo | null>(
    $info.info?.backends.find((b) => b.name === backend) ?? null
  );

  // ``respect EOS`` is locked only for cloud backends that DON'T advertise
  // the Fireworks-style ``ignore_eos`` field. Fireworks unlocks this
  // checkbox (we ship ``ignore_eos: true`` on the wire); NIM, OpenRouter,
  // and LM Studio still leave it pinned because they have no documented
  // escape hatch -- the upstream silently halts on EOS no matter what.
  let respectEosLocked = $derived<boolean>(
    backendInfo?.family === 'cloud' && !backendInfo?.capabilities?.supports_ignore_eos
  );
  $effect(() => {
    if (respectEosLocked && !respectEos) respectEos = true;
  });
  // Show the service-tier selector only on backends that advertise it
  // (Fireworks). Other backends silently ignore the field anyway, but
  // showing a knob that does nothing is bad UX.
  let serviceTierSupported = $derived<boolean>(
    !!backendInfo?.capabilities?.supports_service_tier
  );
  // Server-timings panel visibility tracks supports_perf_metrics; we only
  // render it when (a) the backend can report metrics and (b) we actually
  // received a non-empty perf frame in the last run.
  let perfPanelSupported = $derived<boolean>(
    !!backendInfo?.capabilities?.supports_perf_metrics
  );
  // ``sampling_mask=count`` tells us how many tokens survived
  // server-side sampling filters at each position. When the backend
  // advertises this capability we surface a dedicated "eligible after
  // filters" column in both prompt-logits and generation-steps tables;
  // otherwise the column would just render "?" everywhere which is
  // worse than not showing it at all.
  let samplingMaskSupported = $derived<boolean>(
    !!backendInfo?.capabilities?.supports_sampling_mask
  );
  let rawOutputSupported = $derived<boolean>(
    !!backendInfo?.capabilities?.supports_raw_output
  );
  let logitBiasSupported = $derived<boolean>(
    !!backendInfo?.capabilities?.supports_logit_bias
  );
  let combinedEchoStreamSupported = $derived<boolean>(
    !!backendInfo?.capabilities?.supports_combined_echo_stream
  );

  // ``alternatives`` ceiling comes from the backend's capabilities. Cloud
  // providers cap aggressively (Fireworks: 5, NIM/OpenRouter: 20,
  // LM Studio: 10); local backends with a real vocab cap much higher
  // (we use 200 as a sane upper bound so the input doesn't become a
  // free-form free-for-all). Without the cap the user could ask for
  // ``top_k=100`` against Fireworks and silently get 5 back -- the
  // exact "silently ignored" UX the user pointed at.
  let altsMax = $derived<number>(backendInfo?.capabilities?.max_top_logprobs ?? 50);
  $effect(() => {
    // Whenever the cap shrinks (backend swap or capabilities refresh),
    // re-clamp the current value. Never grow it automatically -- if
    // the user picked 3 and the cap is 20, leave it at 3.
    if (alternatives > altsMax) alternatives = altsMax;
    if (alternatives < 1) alternatives = 1;
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
    backendInfo?.capabilities?.prompt_logprobs === false
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

  async function run() {
    streamError = null;
    steps = [];
    promptSteps = [];
    promptNote = '';
    stopReason = null;
    usage = null;
    perf = null;
    rawOutput = null;
    busy = true;
    const stop_ids = stopIds
      .map((s) => Number.parseInt(s, 10))
      .filter((n) => Number.isFinite(n));
    const stream = apiStream('/api/v1/generate/stream', {
      backend,
      model: model || undefined,
      prompt,
      sampler: { name: sampler, params: samplerParams() },
      max_tokens: maxTokens,
      // Single source of truth for "how many top alternatives do we
      // care about?" -- both the wire ``top_k`` and the table renderer
      // below read from the same state, capped to the backend's
      // ``max_top_logprobs`` ceiling.
      top_k: alternatives,
      stop_texts: stopTexts,
      stop_ids,
      seed,
      respect_eos: respectEos,
      include_prompt: includePrompt,
      // Only ship the service tier when the backend can honor it; the
      // middleware ignores the field otherwise, but keeping the wire
      // small for non-supporting backends is good hygiene.
      service_tier: serviceTierSupported ? serviceTier : undefined,
      logit_bias: collectLogitBias(),
      echo_last: includePrompt && combinedEchoStreamSupported && echoLast > 0 ? echoLast : undefined
    });
    cancelFn = stream.cancel;
    try {
      for await (const evt of stream.events) {
        if (evt.event === 'step') {
          steps = [...steps, evt.step as GenStep];
        } else if (evt.event === 'prompt_score') {
          promptSteps = (evt as { steps: StepResult[] }).steps ?? [];
          promptNote = (evt as { note?: string }).note ?? '';
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
</script>

<Toast message={streamError} onClose={() => (streamError = null)} />

<div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
  <div class="card lg:col-span-1 space-y-3">
    <h2 class="text-lg font-semibold">Generate</h2>
    <p class="text-xs text-slate-400">
      Stream tokens with a chosen sampler. Mirrors <span class="font-mono">dsbx generate</span>.
    </p>
    <BackendSelect bind:value={backend} onChange={onBackendChange} />
    <ModelInput backend={backendInfo} bind:value={model} />
    <CapabilityBadges backend={backend} />
    <div>
      <label class="label" for="prompt">Prompt</label>
      <textarea id="prompt" rows="3" class="input font-mono" bind:value={prompt}></textarea>
    </div>
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
          <div class="space-y-1">
            {#each logitBiasRows as row (row.id)}
              <div class="flex items-center gap-2">
                <input
                  type="number"
                  class="input flex-1 font-mono text-xs"
                  placeholder="token_id"
                  bind:value={row.tokenId}
                />
                <input
                  type="number"
                  step="0.5"
                  min="-100"
                  max="100"
                  class="input w-24 font-mono text-xs"
                  placeholder="bias"
                  bind:value={row.bias}
                />
                <button
                  type="button"
                  class="text-xs px-2 py-0.5 rounded border border-slate-700 hover:border-rose-500 hover:text-rose-300"
                  title="remove this entry"
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
            no entries -- click <em>add</em> to bias a token (e.g. -100 to ban, +5 to nudge past top_p).
          </div>
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
      </div>
    {/if}
    <div class="flex gap-2">
      <button class="btn btn-primary flex-1" onclick={run} disabled={busy || !backend}>
        {busy ? 'streaming…' : 'generate'}
      </button>
      {#if busy}
        <button class="btn btn-ghost" onclick={cancel}>stop</button>
      {/if}
    </div>
  </div>

  <div class="lg:col-span-2 space-y-3">
    <div class="card">
      <div class="flex items-center justify-between mb-1">
        <div class="text-xs uppercase tracking-wider text-slate-500">running completion</div>
        <div class="text-[10px] uppercase tracking-wider text-slate-600 flex items-center gap-2">
          <span class="inline-block w-3 h-3 rounded bg-emerald-500/40"></span>≥80%
          <span class="inline-block w-3 h-3 rounded bg-sky-500/40"></span>≥50%
          <span class="inline-block w-3 h-3 rounded bg-amber-500/40"></span>≥25%
          <span class="inline-block w-3 h-3 rounded bg-orange-500/40"></span>≥10%
          <span class="inline-block w-3 h-3 rounded bg-rose-500/40"></span>&lt;10%
        </div>
      </div>
      <div class="font-mono text-sm text-slate-200 whitespace-pre-wrap min-h-[2.5rem] leading-relaxed">
        {#if promptHasFullTokenization}{#each promptPrefixSteps as ps}{@const lp = ps.chosen?.logprob ?? null}{@const p = probFromLogprob(lp)}<TokenInline
              text={ps.chosen?.text ?? ''}
              isSpecial={ps.chosen?.is_special ?? false}
              showMarkers={showMarkers}
              bgClass={tokenBackgroundClass(p)}
              title={`prompt · p=${p !== null ? ((p ?? 0) * 100).toFixed(2) + '%' : '?'}`}
            />{/each}{:else}<span class="text-slate-400">{prompt}</span>{/if}{#each steps as s}<TokenInline
            text={s.decision.token_text}
            showMarkers={showMarkers}
            bgClass={tokenBackgroundClass(chosenProb(s))}
            title={`p=${chosenProb(s) !== null ? ((chosenProb(s) ?? 0) * 100).toFixed(2) + '%' : '?'}`}
          />{/each}{#if busy}<span class="animate-pulse text-sky-400">▌</span>{/if}
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
                  title="Number of tokens that survived server-side sampling filters (sampling_mask=count). Lower means a more constrained next-token distribution."
                  >eligible</th
                >
              {/if}
              <th class="table-cell text-left">top alts (rank 1..{alternatives})</th>
            </tr>
          </thead>
          <tbody>
            {#each promptSteps as s}
              {@const chosenLP = s.chosen?.logprob ?? null}
              {@const chosenP = probFromLogprob(chosenLP)}
              <tr class="border-b border-slate-800/60">
                <td class="table-cell font-mono text-slate-400">{s.position}</td>
                <td class="table-cell font-mono text-slate-500 text-xs">
                  {#if s.context_text !== null && s.context_text !== undefined}
                    <TokenText text={s.context_text} className="font-mono text-xs" />
                  {/if}
                </td>
                <td class="table-cell">
                  {#if s.chosen}
                    <TokenText text={s.chosen.text} isSpecial={s.chosen.is_special} className="font-mono" />
                  {:else}
                    <span class="text-slate-500">?</span>
                  {/if}
                </td>
                <td class="table-cell w-40"><ConfidenceBar prob={chosenP} /></td>
                {#if samplingMaskSupported}
                  <td class="table-cell font-mono text-xs text-slate-400 tabular-nums">
                    {s.candidates[0]?.sampling_mask_count ?? '?'}
                  </td>
                {/if}
                <td class="table-cell">
                  <div class="flex flex-col gap-0.5">
                    {#each s.candidates.slice(0, alternatives) as c}
                      <div class="flex items-center gap-2">
                        <TokenText text={c.text} isSpecial={c.is_special} className="font-mono text-xs" />
                        <span class="text-xs text-slate-500 tabular-nums">
                          {c.logprob !== null ? (Math.exp(c.logprob) * 100).toFixed(1) + '%' : '?'}
                        </span>
                      </div>
                    {/each}
                  </div>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    {#if steps.length}
      <div class="card overflow-x-auto">
        <div class="text-xs uppercase tracking-wider text-slate-500 mb-2">generation steps</div>
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
                  >eligible</th
                >
              {/if}
              <th class="table-cell text-left">top alts (rank 1..{alternatives})</th>
            </tr>
          </thead>
          <tbody>
            {#each steps as s}
              <tr class="border-b border-slate-800/60" data-token-row>
                <td class="table-cell font-mono text-slate-400">{s.step}</td>
                <td class="table-cell">
                  <TokenText text={s.decision.token_text} className="font-mono" />
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
                    {#each s.step_result.candidates.slice(0, alternatives) as c}
                      <div class="flex items-center gap-2">
                        <TokenText text={c.text} isSpecial={c.is_special} className="font-mono text-xs" />
                        <span class="text-xs text-slate-500 tabular-nums">
                          {c.logprob !== null ? (Math.exp(c.logprob) * 100).toFixed(1) + '%' : '?'}
                        </span>
                      </div>
                    {/each}
                  </div>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>
</div>
