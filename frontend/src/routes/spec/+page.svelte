<script lang="ts">
  import { onMount } from 'svelte';
  import BackendSelect from '$lib/components/BackendSelect.svelte';
  import TokenText from '$lib/components/TokenText.svelte';
  import Toast from '$lib/components/Toast.svelte';
  import { apiStream } from '$lib/api';
  import { info } from '$lib/stores/info';
  import type { SpecRound, TokenCandidate } from '$lib/types';

  let targetBackend = $state('');
  let draftBackend = $state('');
  let prompt = $state('The capital of France is');
  let gamma = $state(4);
  let maxTokens = $state(24);
  let rounds = $state<SpecRound[]>([]);
  let busy = $state(false);
  let error = $state<string | null>(null);
  let summary = $state<{
    total_proposed: number;
    total_accepted: number;
    total_emitted: number;
    completion: string;
  } | null>(null);
  let cancelFn: (() => void) | null = null;

  onMount(async () => {
    if (!$info.info) await info.refresh();
    const localHf = $info.info?.backends.find((b) => b.name === 'hf')?.name;
    targetBackend = localHf ?? $info.info?.default_backend ?? '';
    draftBackend = $info.info?.backends.find((b) => b.name !== targetBackend)?.name ?? '';
  });

  async function run() {
    error = null;
    rounds = [];
    summary = null;
    busy = true;
    const stream = apiStream('/api/v1/spec/stream', {
      target_backend: targetBackend,
      draft_backend: draftBackend,
      prompt,
      gamma,
      max_tokens: maxTokens
    });
    cancelFn = stream.cancel;
    try {
      for await (const evt of stream.events) {
        if (evt.event === 'round') {
          rounds = [...rounds, (evt as { round: SpecRound }).round];
        } else if (evt.event === 'done') {
          summary = {
            total_proposed: (evt as { total_proposed?: number }).total_proposed ?? 0,
            total_accepted: (evt as { total_accepted?: number }).total_accepted ?? 0,
            total_emitted: (evt as { total_emitted?: number }).total_emitted ?? 0,
            completion: (evt as { completion?: string }).completion ?? ''
          };
          const err = (evt as { error?: string | null }).error;
          if (err) error = err;
        }
      }
    } catch (exc) {
      error = exc instanceof Error ? exc.message : String(exc);
    } finally {
      busy = false;
      cancelFn = null;
    }
  }
</script>

<Toast message={error} onClose={() => (error = null)} />

<div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
  <div class="card lg:col-span-1 space-y-3">
    <h2 class="text-lg font-semibold">Speculative decoding</h2>
    <p class="text-xs text-slate-400">
      Draft proposes <span class="font-mono">gamma</span> tokens greedily; target verifies in
      one pass. Accepted prefix coloured green; rejected drafts in red. Needs an HF target
      (with <span class="font-mono">verify_greedy</span>) and any draft sharing its tokenizer.
    </p>
    <BackendSelect
      bind:value={targetBackend}
      onChange={(v) => (targetBackend = v)}
      label="Target"
      id="target"
    />
    <BackendSelect
      bind:value={draftBackend}
      onChange={(v) => (draftBackend = v)}
      label="Draft"
      id="draft"
    />
    <div>
      <label class="label" for="prompt">Prompt</label>
      <textarea id="prompt" rows="3" class="input font-mono" bind:value={prompt}></textarea>
    </div>
    <div class="grid grid-cols-2 gap-2">
      <div>
        <label class="label" for="g">gamma</label>
        <input id="g" type="number" min="1" max="16" class="input" bind:value={gamma} />
      </div>
      <div>
        <label class="label" for="mt">max tokens</label>
        <input id="mt" type="number" min="1" max="200" class="input" bind:value={maxTokens} />
      </div>
    </div>
    <button class="btn btn-primary w-full" onclick={run} disabled={busy || !targetBackend || !draftBackend}>
      {busy ? 'streaming…' : 'speculate'}
    </button>
  </div>

  <div class="lg:col-span-2 space-y-3">
    {#if summary}
      <div class="card text-sm">
        <div class="font-mono text-slate-200 whitespace-pre-wrap">
          {prompt}<span class="text-sky-300">{summary.completion}</span>
        </div>
        <div class="text-xs text-slate-500 mt-2">
          {summary.total_accepted} of {summary.total_proposed} drafts accepted
          → {summary.total_emitted} tokens in {rounds.length} round(s).
        </div>
      </div>
    {/if}
    {#if rounds.length}
      <div class="card space-y-2">
        {#each rounds as r}
          <div class="flex items-start gap-2">
            <span class="text-xs font-mono text-slate-500 w-10">r{r.step}</span>
            <div class="flex flex-wrap gap-1">
              {#each r.proposed as p, i}
                {@const accepted = i < r.accepted}
                <span
                  class="chip {accepted
                    ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-200'
                    : 'bg-rose-500/20 border-rose-500/40 text-rose-200'}"
                  title={accepted ? 'accepted by target' : 'rejected'}
                >
                  <TokenText text={p.text} isSpecial={p.is_special} className="font-mono text-xs" />
                </span>
              {/each}
              {#if r.correction}
                <span class="chip bg-sky-500/20 border-sky-500/40 text-sky-200" title="target correction / bonus">
                  +<TokenText text={r.correction.text} isSpecial={r.correction.is_special} className="font-mono text-xs" />
                </span>
              {/if}
            </div>
          </div>
        {/each}
      </div>
    {:else if !summary}
      <div class="card text-center text-slate-500 text-sm">
        Pick a target + draft, hit speculate. Both backends must share a tokenizer
        (typically two HF models from the same family).
      </div>
    {/if}
  </div>
</div>
