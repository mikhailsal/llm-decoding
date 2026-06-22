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
  import { ApiError, apiFetch } from '$lib/api';
  import { info } from '$lib/stores/info';
  import { probFromLogprob, tokenBackgroundClass } from '$lib/render';
  import type {
    BackendInfo,
    InspectResponse,
    StepResult,
    Watched,
    TokenCandidate
  } from '$lib/types';

  let backend = $state<string>('');
  let model = $state<string>('');
  let prompt = $state('The capital of France is Paris');
  // Single ``alternatives`` knob (see /generate for the rationale: a
  // separate "fetched" / "shown" pair invited a confusing failure
  // mode where the "shown" axis silently rendered empty rows past
  // the provider-capped "fetched" value).
  let alternatives = $state(5);
  let watchTexts = $state<string[]>([]);
  let watchIds = $state<string[]>([]);
  let watchEos = $state(false);
  let showMarkers = $state(true);
  let busy = $state(false);
  let result = $state<InspectResponse | null>(null);
  let error = $state<string | null>(null);

  let backendInfo = $derived<BackendInfo | null>(
    $info.info?.backends.find((b) => b.name === backend) ?? null
  );

  // Cap ``alternatives`` to whatever the active backend can actually
  // return. Fireworks tops out at 5, NIM/OpenRouter at 20, LM Studio
  // at 10; local backends with a real vocab go much higher. Without
  // this, asking for 50 against Fireworks silently returned 5 and
  // rendered 45 empty rows -- the exact "silently ignored" UX the
  // dual-knob audit flagged.
  let altsMax = $derived<number>(backendInfo?.capabilities?.max_top_logprobs ?? 50);
  $effect(() => {
    if (alternatives > altsMax) alternatives = altsMax;
    if (alternatives < 1) alternatives = 1;
  });

  onMount(async () => {
    if (!$info.info) await info.refresh();
    backend = $info.info?.default_backend ?? '';
    model = backendInfo?.loaded_model ?? '';
  });

  function onBackendChange(next: string) {
    info.select(next);
    const b = $info.info?.backends.find((x) => x.name === next) ?? null;
    model = b?.loaded_model ?? '';
  }

  function watchedById(step: StepResult, id: number): TokenCandidate | null {
    const w = step.watched.find((x: Watched) => x.token_id === id);
    return w ? w.candidate : null;
  }

  async function run() {
    error = null;
    result = null;
    busy = true;
    try {
      const ids = watchIds
        .map((s) => Number.parseInt(s, 10))
        .filter((n) => Number.isFinite(n));
      const data = await apiFetch<InspectResponse>('/api/v1/inspect', {
        method: 'POST',
        body: JSON.stringify({
          backend,
          model: model || undefined,
          prompt,
          top_k: alternatives,
          watch_texts: watchTexts,
          watch_ids: ids,
          watch_eos: watchEos
        })
      });
      result = data;
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    } finally {
      busy = false;
    }
  }
</script>

<Toast message={error} onClose={() => (error = null)} />

<div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
  <!-- Controls -->
  <div class="card lg:col-span-1 space-y-3">
    <h2 class="text-lg font-semibold">Inspect</h2>
    <p class="text-xs text-slate-400">
      Score the next-token distribution at every position of the prompt.
      Mirrors <span class="font-mono">dsbx inspect</span>.
    </p>
    <BackendSelect bind:value={backend} onChange={onBackendChange} />
    <ModelInput backend={backendInfo} bind:value={model} />
    <CapabilityBadges backend={backend} />
    <div>
      <label class="label" for="prompt">Prompt</label>
      <textarea
        id="prompt"
        rows="3"
        class="input font-mono"
        bind:value={prompt}
      ></textarea>
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
    <ChipInput
      bind:values={watchTexts}
      label="Watch text"
      placeholder="e.g. ' Paris' (leading space preserved)"
      hint="Each chip is tokenized; multi-token watches use their first id."
    />
    <ChipInput
      bind:values={watchIds}
      label="Watch ids"
      placeholder="numeric token id"
      preserveSpace={false}
    />
    <div class="space-y-2">
      <label class="flex items-center gap-2 text-sm">
        <input type="checkbox" bind:checked={watchEos} class="accent-sky-500" />
        Watch EOS tokens
      </label>
      <label class="flex items-center gap-2 text-sm">
        <input type="checkbox" bind:checked={showMarkers} class="accent-sky-500" />
        show whitespace markers (<span class="font-mono">␣ ↵ →</span>)
      </label>
    </div>
    <button class="btn btn-primary w-full" onclick={run} disabled={busy || !backend || !prompt}>
      {busy ? 'inspecting…' : 'run inspect'}
    </button>
  </div>

  <!-- Output -->
  <div class="lg:col-span-2 space-y-3">
    {#if result?.note}
      <div class="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-md p-2">
        {result.note}
      </div>
    {/if}
    {#if result}
      <div class="card">
        <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">
          prompt as scored (background = chosen-token confidence)
        </div>
        <div class="font-mono text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
          {#each result.steps as step}
            {@const p = probFromLogprob(step.chosen?.logprob ?? null)}
            {#if step.chosen}
              <TokenInline
                text={step.chosen.text}
                isSpecial={step.chosen.is_special}
                showMarkers={showMarkers}
                bgClass={tokenBackgroundClass(p)}
                title={`pos ${step.position} · p=${p !== null ? (p * 100).toFixed(2) + '%' : '?'}`}
              />
            {/if}
          {/each}
        </div>
      </div>
      <div class="card overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-xs text-slate-400 border-b border-slate-800">
            <tr>
              <th class="table-cell text-left">pos</th>
              <th class="table-cell text-left">token chosen</th>
              <th class="table-cell text-left">prob</th>
              <th class="table-cell text-left">top candidates (rank 1..{alternatives})</th>
              {#each result.watches as w}
                <th class="table-cell text-left">{w.label}</th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each result.steps as step}
              {@const chosenP = probFromLogprob(step.chosen?.logprob ?? null)}
              <tr class="border-b border-slate-800/60 align-top" data-token-row>
                <td class="table-cell font-mono text-slate-400">{step.position}</td>
                <td class="table-cell">
                  {#if step.chosen}
                    <TokenText text={step.chosen.text} isSpecial={step.chosen.is_special} />
                  {:else}
                    <span class="text-slate-500 text-xs">(predict next)</span>
                  {/if}
                </td>
                <td class="table-cell w-40"><ConfidenceBar prob={chosenP} /></td>
                <td class="table-cell">
                  <div class="space-y-0.5">
                    {#each step.candidates.slice(0, alternatives) as c, j}
                      <div class="flex items-center gap-2">
                        <span class="text-xs text-slate-500 font-mono w-6 text-right">{j + 1}.</span>
                        <TokenText text={c.text} isSpecial={c.is_special} className="font-mono text-xs" />
                        <ConfidenceBar prob={c.logprob !== null ? Math.exp(c.logprob) : null} />
                      </div>
                    {/each}
                  </div>
                </td>
                {#each result.watches as w}
                  {@const cand = watchedById(step, w.token_id)}
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
    {:else}
      <div class="card text-center text-slate-500 text-sm">
        Pick a backend, enter a prompt, hit “run inspect”. The chosen-per-step
        column shows what the model would have emitted; the right columns are
        your watch tokens.
      </div>
    {/if}
  </div>
</div>
