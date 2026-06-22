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
  import type { GenStep, StepResult, TokenCandidate, BackendInfo } from '$lib/types';

  let backend = $state<string>('');
  let model = $state<string>('');
  let prompt = $state('Once upon a time');
  let maxTokens = $state(20);
  let topK = $state(8);
  let altCount = $state(3);
  let seed = $state(0);
  let stopTexts = $state<string[]>([]);
  let stopIds = $state<string[]>([]);
  let respectEos = $state(true);
  let includePrompt = $state(false);
  let showMarkers = $state(true);

  type SamplerName = 'greedy' | 'temperature' | 'top_k' | 'top_p' | 'min_p' | 'typical';
  let sampler = $state<SamplerName>('greedy');
  let temperature = $state(1.0);
  let samplerTopK = $state(40);
  let topP = $state(0.9);
  let minP = $state(0.05);
  let typicalP = $state(0.95);

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

  let backendInfo = $derived<BackendInfo | null>(
    $info.info?.backends.find((b) => b.name === backend) ?? null
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
    switch (sampler) {
      case 'greedy':
        return {};
      case 'temperature':
        return { temperature };
      case 'top_k':
        return { temperature, top_k: samplerTopK };
      case 'top_p':
        return { temperature, top_p: topP };
      case 'min_p':
        return { temperature, min_p: minP };
      case 'typical':
        return { temperature, typical_p: typicalP };
    }
  }

  async function run() {
    streamError = null;
    steps = [];
    promptSteps = [];
    promptNote = '';
    stopReason = null;
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
      top_k: topK,
      stop_texts: stopTexts,
      stop_ids,
      seed,
      respect_eos: respectEos,
      include_prompt: includePrompt
    });
    cancelFn = stream.cancel;
    try {
      for await (const evt of stream.events) {
        if (evt.event === 'step') {
          steps = [...steps, evt.step as GenStep];
        } else if (evt.event === 'prompt_score') {
          promptSteps = (evt as { steps: StepResult[] }).steps ?? [];
          promptNote = (evt as { note?: string }).note ?? '';
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
        <label class="label" for="mt">max</label>
        <input id="mt" type="number" min="1" max="200" class="input" bind:value={maxTokens} />
      </div>
      <div>
        <label class="label" for="tk">top_k (fetched)</label>
        <input id="tk" type="number" min="1" max="200" class="input" bind:value={topK} />
      </div>
      <div>
        <label class="label" for="alt">alternatives shown</label>
        <input id="alt" type="number" min="1" max="50" class="input" bind:value={altCount} />
      </div>
      <div>
        <label class="label" for="seed">seed</label>
        <input id="seed" type="number" class="input" bind:value={seed} />
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
      </div>
    {/if}
    <ChipInput bind:values={stopTexts} label="Stop text" placeholder="e.g. '.'" />
    <ChipInput bind:values={stopIds} label="Stop ids" placeholder="token id" preserveSpace={false} />
    <div class="space-y-2">
      <label class="flex items-center gap-2 text-sm">
        <input type="checkbox" bind:checked={respectEos} class="accent-sky-500" />
        respect EOS
      </label>
      <label class="flex items-center gap-2 text-sm">
        <input type="checkbox" bind:checked={includePrompt} class="accent-sky-500" />
        include prompt logits
      </label>
      <label class="flex items-center gap-2 text-sm">
        <input type="checkbox" bind:checked={showMarkers} class="accent-sky-500" />
        show whitespace markers (<span class="font-mono">␣ ↵ →</span>) in completion
      </label>
    </div>
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
        <span class="text-slate-400">{prompt}</span>{#each steps as s}<TokenInline
            text={s.decision.token_text}
            showMarkers={showMarkers}
            bgClass={tokenBackgroundClass(chosenProb(s))}
            title={`p=${chosenProb(s) !== null ? ((chosenProb(s) ?? 0) * 100).toFixed(2) + '%' : '?'}`}
          />{/each}{#if busy}<span class="animate-pulse text-sky-400">▌</span>{/if}
      </div>
      {#if stopReason}
        <div class="text-xs text-slate-500 mt-2">stopped: <span class="font-mono">{stopReason}</span></div>
      {/if}
    </div>

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
              <th class="table-cell text-left">top alts (rank 1..{altCount})</th>
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
                <td class="table-cell">
                  <div class="flex flex-col gap-0.5">
                    {#each s.candidates.slice(0, altCount) as c}
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
              <th class="table-cell text-left">top alts (rank 1..{altCount})</th>
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
                <td class="table-cell">
                  <div class="flex flex-col gap-0.5">
                    {#each s.step_result.candidates.slice(0, altCount) as c}
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
