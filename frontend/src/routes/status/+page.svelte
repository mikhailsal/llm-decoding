<script lang="ts">
  import { onMount } from 'svelte';
  import Toast from '$lib/components/Toast.svelte';
  import { ApiError, apiFetch } from '$lib/api';
  import { info } from '$lib/stores/info';
  import type { ProbeResponse } from '$lib/types';

  let probe = $state<ProbeResponse | null>(null);
  let probeBusy = $state(false);
  let error = $state<string | null>(null);

  async function refreshProbe(force = false) {
    probeBusy = true;
    try {
      probe = await apiFetch<ProbeResponse>(`/api/v1/probe${force ? '?refresh=true' : ''}`);
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    } finally {
      probeBusy = false;
    }
  }

  onMount(async () => {
    if (!$info.info) await info.refresh();
    refreshProbe(false);
  });
</script>

<Toast message={error} onClose={() => (error = null)} />

<div class="space-y-4">
  <div class="card">
    <h2 class="text-lg font-semibold mb-3">Backends</h2>
    {#if $info.info}
      <div class="text-xs text-slate-500 mb-2">
        engine {$info.info.engine_version} · default
        <span class="font-mono text-slate-300">{$info.info.default_backend}</span>
      </div>
      <table class="w-full text-sm">
        <thead class="text-xs text-slate-400 border-b border-slate-800">
          <tr>
            <th class="table-cell text-left">name</th>
            <th class="table-cell text-left">family</th>
            <th class="table-cell text-left">model</th>
            <th class="table-cell text-left">status</th>
            <th class="table-cell text-left">capabilities</th>
          </tr>
        </thead>
        <tbody>
          {#each $info.info.backends as b}
            <tr class="border-b border-slate-800/60 align-top">
              <td class="table-cell font-mono">{b.name}</td>
              <td class="table-cell text-slate-400">{b.family}</td>
              <td class="table-cell text-xs">
                {#if b.loaded_model}
                  <span class="font-mono text-slate-300">{b.loaded_model}</span>
                  {#if b.model_editable}
                    <span class="text-slate-500"> · editable</span>
                  {/if}
                {:else}
                  <span class="text-slate-600">unknown</span>
                {/if}
                {#if b.suggested_models.length > 1}
                  <div class="text-[10px] text-slate-500 mt-0.5 font-mono">
                    +{b.suggested_models.length - 1} more in picker
                  </div>
                {/if}
              </td>
              <td class="table-cell">
                {#if b.available}
                  <span class="text-emerald-300">available</span>
                {:else}
                  <span class="text-amber-400">unavailable</span>
                  {#if b.note}
                    <span class="text-xs text-slate-500 font-mono"> — {b.note}</span>
                  {/if}
                {/if}
              </td>
              <td class="table-cell text-xs text-slate-400">
                {#if b.capabilities}
                  top_k≤{b.capabilities.max_top_logprobs}
                  {#if b.capabilities.full_vocab} · full-vocab{/if}
                  {#if b.capabilities.prompt_logprobs} · prompt-lp{/if}
                  {#if b.capabilities.can_force_token} · force{/if}
                {:else}
                  <span class="text-slate-600">(not yet loaded)</span>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>

  <div class="card">
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-lg font-semibold">Cloud provider probe</h2>
      <div class="flex items-center gap-2">
        {#if probe?.fresh === false && probe?.cached_at}
          <span class="text-xs text-slate-500">
            cached {new Date(probe.cached_at * 1000).toLocaleTimeString()}
          </span>
        {/if}
        <button class="btn btn-ghost text-xs" onclick={() => refreshProbe(true)} disabled={probeBusy}>
          {probeBusy ? 'probing…' : 'refresh'}
        </button>
      </div>
    </div>
    {#if probe}
      <table class="w-full text-sm">
        <thead class="text-xs text-slate-400 border-b border-slate-800">
          <tr>
            <th class="table-cell text-left">provider</th>
            <th class="table-cell text-left">model</th>
            <th class="table-cell text-left">chat logprobs</th>
            <th class="table-cell text-left">prompt logprobs</th>
          </tr>
        </thead>
        <tbody>
          {#each probe.rows as row}
            <tr class="border-b border-slate-800/60">
              <td class="table-cell font-mono">{row.provider}</td>
              <td class="table-cell text-slate-400 font-mono text-xs">{row.model}</td>
              <td class="table-cell font-mono text-xs">{row.chat_logprobs}</td>
              <td class="table-cell font-mono text-xs">{row.prompt_logprobs}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>
</div>
