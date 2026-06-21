<script lang="ts">
  import { onMount } from 'svelte';
  import BackendSelect from '$lib/components/BackendSelect.svelte';
  import CapabilityBadges from '$lib/components/CapabilityBadges.svelte';
  import ChipInput from '$lib/components/ChipInput.svelte';
  import ConfidenceBar from '$lib/components/ConfidenceBar.svelte';
  import TokenText from '$lib/components/TokenText.svelte';
  import Toast from '$lib/components/Toast.svelte';
  import { apiStream } from '$lib/api';
  import { info } from '$lib/stores/info';
  import type { GenStep, TokenCandidate } from '$lib/types';

  let backend = $state<string>('');
  let prompt = $state('Once upon a time');
  let maxTokens = $state(20);
  let topK = $state(8);
  let seed = $state(0);
  let stopTexts = $state<string[]>([]);
  let stopIds = $state<string[]>([]);
  let respectEos = $state(true);

  type SamplerName = 'greedy' | 'temperature' | 'top_k' | 'top_p' | 'min_p' | 'typical';
  let sampler = $state<SamplerName>('greedy');
  let temperature = $state(1.0);
  let samplerTopK = $state(40);
  let topP = $state(0.9);
  let minP = $state(0.05);
  let typicalP = $state(0.95);

  let steps = $state<GenStep[]>([]);
  let stopReason = $state<string | null>(null);
  let streamError = $state<string | null>(null);
  let busy = $state(false);
  let cancelFn: (() => void) | null = null;

  onMount(async () => {
    if (!$info.info) await info.refresh();
    backend = $info.info?.default_backend ?? '';
  });

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
    stopReason = null;
    busy = true;
    const stop_ids = stopIds
      .map((s) => Number.parseInt(s, 10))
      .filter((n) => Number.isFinite(n));
    const stream = apiStream('/api/v1/generate/stream', {
      backend,
      prompt,
      sampler: { name: sampler, params: samplerParams() },
      max_tokens: maxTokens,
      top_k: topK,
      stop_texts: stopTexts,
      stop_ids,
      seed,
      respect_eos: respectEos
    });
    cancelFn = stream.cancel;
    try {
      for await (const evt of stream.events) {
        if (evt.event === 'step') {
          steps = [...steps, evt.step as GenStep];
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

  let generatedText = $derived(steps.map((s) => s.decision.token_text).join(''));
  let chosenProb = (step: GenStep): number | null => {
    const id = step.decision.token_id;
    const c = step.step_result.candidates.find((c: TokenCandidate) => c.token_id === id);
    if (!c || c.logprob === null) return null;
    return Math.exp(c.logprob);
  };
</script>

<Toast message={streamError} onClose={() => (streamError = null)} />

<div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
  <div class="card lg:col-span-1 space-y-3">
    <h2 class="text-lg font-semibold">Generate</h2>
    <p class="text-xs text-slate-400">
      Stream tokens with a chosen sampler. Mirrors <span class="font-mono">dsbx generate</span>.
    </p>
    <BackendSelect bind:value={backend} onChange={(v) => info.select(v)} />
    <CapabilityBadges backend={backend} />
    <div>
      <label class="label" for="prompt">Prompt</label>
      <textarea id="prompt" rows="3" class="input font-mono" bind:value={prompt}></textarea>
    </div>
    <div class="grid grid-cols-3 gap-2">
      <div>
        <label class="label" for="mt">max</label>
        <input id="mt" type="number" min="1" max="200" class="input" bind:value={maxTokens} />
      </div>
      <div>
        <label class="label" for="tk">top_k</label>
        <input id="tk" type="number" min="1" max="200" class="input" bind:value={topK} />
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
    <label class="flex items-center gap-2 text-sm">
      <input type="checkbox" bind:checked={respectEos} class="accent-sky-500" />
      respect EOS
    </label>
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
      <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">running completion</div>
      <div class="font-mono text-sm text-slate-200 whitespace-pre-wrap min-h-[2.5rem]">
        {prompt}<span class="text-sky-300">{generatedText}</span>
        {#if busy}<span class="animate-pulse text-sky-400">▌</span>{/if}
      </div>
      {#if stopReason}
        <div class="text-xs text-slate-500 mt-2">stopped: <span class="font-mono">{stopReason}</span></div>
      {/if}
    </div>

    {#if steps.length}
      <div class="card overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-xs text-slate-400 border-b border-slate-800">
            <tr>
              <th class="table-cell text-left">step</th>
              <th class="table-cell text-left">chosen</th>
              <th class="table-cell text-left">prob</th>
              <th class="table-cell text-left">top alts (rank 1..3)</th>
              <th class="table-cell text-left">sampler note</th>
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
                    {#each s.step_result.candidates.slice(0, 3) as c}
                      <div class="flex items-center gap-2">
                        <TokenText text={c.text} isSpecial={c.is_special} className="font-mono text-xs" />
                        <span class="text-xs text-slate-500 tabular-nums">
                          {c.logprob !== null ? (Math.exp(c.logprob) * 100).toFixed(1) + '%' : '?'}
                        </span>
                      </div>
                    {/each}
                  </div>
                </td>
                <td class="table-cell text-xs font-mono text-slate-400">
                  {s.decision.note}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>
</div>
