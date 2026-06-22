<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import BackendSelect from '$lib/components/BackendSelect.svelte';
  import ModelInput from '$lib/components/ModelInput.svelte';
  import CapabilityBadges from '$lib/components/CapabilityBadges.svelte';
  import ConfidenceBar from '$lib/components/ConfidenceBar.svelte';
  import TokenText from '$lib/components/TokenText.svelte';
  import TokenInline from '$lib/components/TokenInline.svelte';
  import Toast from '$lib/components/Toast.svelte';
  import { ApiError, apiFetch } from '$lib/api';
  import { info } from '$lib/stores/info';
  import { tokenBackgroundClass } from '$lib/render';
  import type { BackendInfo, ManualSnapshot, ManualTranscript } from '$lib/types';

  let backend = $state<string>('');
  let model = $state<string>('');
  let prompt = $state('The capital of France is');
  let topK = $state(8);
  let showMarkers = $state(true);
  let snap = $state<ManualSnapshot | null>(null);
  let error = $state<string | null>(null);
  let busy = $state(false);
  let forceText = $state('');
  let forceId = $state('');

  let backendInfo = $derived<BackendInfo | null>(
    $info.info?.backends.find((b) => b.name === backend) ?? null
  );

  onMount(async () => {
    if (!$info.info) await info.refresh();
    backend = $info.info?.default_backend ?? '';
    model = backendInfo?.loaded_model ?? '';
    window.addEventListener('keydown', onGlobalKey);
  });
  onDestroy(() => window.removeEventListener('keydown', onGlobalKey));

  function onBackendChange(next: string) {
    info.select(next);
    const b = $info.info?.backends.find((x) => x.name === next) ?? null;
    model = b?.loaded_model ?? '';
  }

  async function create() {
    error = null;
    try {
      busy = true;
      snap = await apiFetch<ManualSnapshot>('/api/v1/manual/sessions', {
        method: 'POST',
        body: JSON.stringify({
          backend,
          model: model || undefined,
          prompt,
          top_k: topK
        })
      });
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    } finally {
      busy = false;
    }
  }

  async function call<T>(path: string, init: RequestInit): Promise<T> {
    busy = true;
    try {
      return await apiFetch<T>(path, init);
    } finally {
      busy = false;
    }
  }

  async function pick(rank: number) {
    if (!snap) return;
    try {
      snap = await call<ManualSnapshot>(`/api/v1/manual/sessions/${snap.session_id}/pick`, {
        method: 'POST',
        body: JSON.stringify({ rank })
      });
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    }
  }

  async function undo() {
    if (!snap) return;
    try {
      snap = await call<ManualSnapshot>(`/api/v1/manual/sessions/${snap.session_id}/undo`, {
        method: 'POST'
      });
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    }
  }

  async function applyForce() {
    if (!snap) return;
    const payload =
      forceId !== ''
        ? { id: Number.parseInt(forceId, 10) }
        : forceText !== ''
          ? { text: forceText }
          : null;
    if (!payload) return;
    try {
      snap = await call<ManualSnapshot>(`/api/v1/manual/sessions/${snap.session_id}/force`, {
        method: 'POST',
        body: JSON.stringify(payload)
      });
      forceText = '';
      forceId = '';
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    }
  }

  async function changeTopK(next: number) {
    if (!snap) return;
    try {
      snap = await call<ManualSnapshot>(
        `/api/v1/manual/sessions/${snap.session_id}/set_top_k`,
        { method: 'POST', body: JSON.stringify({ top_k: next }) }
      );
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    }
  }

  async function save() {
    if (!snap) return;
    try {
      const data = await apiFetch<ManualTranscript>(
        `/api/v1/manual/sessions/${snap.session_id}/transcript`
      );
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dsbx-manual-${snap.session_id.slice(0, 8)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    }
  }

  async function load(fileEvent: Event) {
    if (!snap) return;
    const f = (fileEvent.target as HTMLInputElement).files?.[0];
    if (!f) return;
    const text = await f.text();
    try {
      const payload = JSON.parse(text) as ManualTranscript;
      snap = await call<ManualSnapshot>(`/api/v1/manual/sessions/${snap.session_id}/load`, {
        method: 'POST',
        body: JSON.stringify(payload)
      });
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    } finally {
      (fileEvent.target as HTMLInputElement).value = '';
    }
  }

  function onGlobalKey(e: KeyboardEvent) {
    if (!snap || busy) return;
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }
    if (e.key >= '1' && e.key <= '9') {
      e.preventDefault();
      pick(Number.parseInt(e.key, 10) - 1);
    } else if (e.key === 'u') {
      e.preventDefault();
      undo();
    } else if (e.key === 's') {
      e.preventDefault();
      save();
    } else if (e.key === 'f' && snap.can_force_token) {
      e.preventDefault();
      document.getElementById('force-text')?.focus();
    }
  }
</script>

<Toast message={error} onClose={() => (error = null)} />

<div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
  <div class="card lg:col-span-1 space-y-3">
    <h2 class="text-lg font-semibold">Manual decoding</h2>
    <p class="text-xs text-slate-400">
      Pick each next token by hand. Keyboard: <span class="font-mono">1..9</span> = pick rank,
      <span class="font-mono">u</span> = undo, <span class="font-mono">f</span> = focus force,
      <span class="font-mono">s</span> = save JSON.
    </p>

    {#if !snap}
      <BackendSelect bind:value={backend} onChange={onBackendChange} />
      <ModelInput backend={backendInfo} bind:value={model} />
      <CapabilityBadges backend={backend} />
      <div>
        <label class="label" for="prompt">Prompt</label>
        <textarea id="prompt" rows="3" class="input font-mono" bind:value={prompt}></textarea>
      </div>
      <div>
        <label class="label" for="tk">top_k</label>
        <input id="tk" type="number" min="1" max="50" class="input w-24" bind:value={topK} />
      </div>
      <label class="flex items-center gap-2 text-sm">
        <input type="checkbox" bind:checked={showMarkers} class="accent-sky-500" />
        show whitespace markers (<span class="font-mono">␣ ↵ →</span>)
      </label>
      <button class="btn btn-primary w-full" onclick={create} disabled={busy || !backend}>
        {busy ? 'starting…' : 'start session'}
      </button>
    {:else}
      <div class="text-xs text-slate-400 font-mono">session {snap.session_id.slice(0, 8)}</div>
      <CapabilityBadges backend={snap.backend} />
      {#if snap.model}
        <div class="text-xs text-slate-500">
          model: <span class="font-mono text-slate-300">{snap.model}</span>
        </div>
      {/if}
      <label class="flex items-center gap-2 text-sm">
        <input type="checkbox" bind:checked={showMarkers} class="accent-sky-500" />
        show whitespace markers (<span class="font-mono">␣ ↵ →</span>)
      </label>
      <div class="grid grid-cols-2 gap-2">
        <button class="btn btn-ghost" onclick={undo} disabled={busy || snap.generated_ids.length === 0}>
          undo (u)
        </button>
        <button class="btn btn-ghost" onclick={save} disabled={busy}>save (s)</button>
      </div>
      <div>
        <label class="label" for="tk2">top_k</label>
        <div class="flex gap-2">
          <input
            id="tk2"
            type="number"
            min="1"
            max="50"
            class="input"
            value={snap.top_k}
            onchange={(e) => changeTopK(Number.parseInt((e.target as HTMLInputElement).value, 10))}
          />
        </div>
      </div>
      {#if snap.can_force_token}
        <div class="border-t border-slate-800 pt-3 space-y-2">
          <div class="text-xs text-slate-500">force token</div>
          <input
            id="force-text"
            type="text"
            class="input font-mono text-sm"
            placeholder="text (e.g. ' however')"
            bind:value={forceText}
          />
          <div class="flex gap-2">
            <input
              type="text"
              class="input font-mono text-sm"
              placeholder="or id"
              bind:value={forceId}
            />
            <button class="btn btn-primary" onclick={applyForce} disabled={busy}>force</button>
          </div>
        </div>
      {:else}
        <div class="text-xs text-slate-500">backend doesn't support force-token.</div>
      {/if}
      <div class="border-t border-slate-800 pt-3">
        <label class="label" for="load">load transcript</label>
        <input id="load" type="file" accept="application/json" class="text-xs" onchange={load} />
      </div>
    {/if}
  </div>

  <div class="lg:col-span-2 space-y-3">
    {#if snap}
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
        <div class="font-mono text-sm whitespace-pre-wrap min-h-[2.5rem] leading-relaxed">
          <span class="text-slate-400">{snap.prompt}</span>{#each snap.generated_pieces as piece, i}{@const p = snap.generated_probs[i] ?? null}<TokenInline
              text={piece}
              showMarkers={showMarkers}
              bgClass={tokenBackgroundClass(p)}
              title={`id ${snap.generated_ids[i]}${p !== null ? ` · p=${(p * 100).toFixed(2)}%` : ' · forced'}`}
            />{/each}
        </div>
      </div>

      <div class="card overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-xs text-slate-400 border-b border-slate-800">
            <tr>
              <th class="table-cell text-left">rank</th>
              <th class="table-cell text-left">token</th>
              <th class="table-cell text-left">probability</th>
              <th class="table-cell text-left">id</th>
            </tr>
          </thead>
          <tbody>
            {#each snap.distribution.candidates as c, i}
              <tr class="border-b border-slate-800/60 cursor-pointer hover:bg-slate-800/40" onclick={() => pick(i)}>
                <td class="table-cell font-mono text-slate-400">{i + 1}.</td>
                <td class="table-cell">
                  <TokenText text={c.text} isSpecial={c.is_special} />
                </td>
                <td class="table-cell w-48">
                  <ConfidenceBar prob={c.logprob !== null ? Math.exp(c.logprob) : null} />
                </td>
                <td class="table-cell font-mono text-xs text-slate-500">{c.token_id}</td>
              </tr>
            {/each}
          </tbody>
        </table>
        <div class="text-xs text-slate-500 mt-2">
          click a row or press its number key to advance.
        </div>
      </div>
    {:else}
      <div class="card text-center text-slate-500 text-sm">
        Start a session on the left to begin decoding by hand.
      </div>
    {/if}
  </div>
</div>
