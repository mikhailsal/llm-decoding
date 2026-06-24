<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { apiFetch, ApiError, getRemoteStatus, reloadRemoteModel } from '$lib/api';
  import { info } from '$lib/stores/info';
  import type { BackendInfo, ModelsResponse, RemoteStatus, RemoteSlotState } from '$lib/types';

  /**
   * Load/reload control for one remote dsbx-serve host's swappable model
   * slot. Shows a live state badge (empty / loading / ready / error), a
   * picker populated from the host's catalogue (``GET /api/v1/models``),
   * and a Load/Reload button. While the slot is ``loading`` it polls
   * ``/status`` every ~2 s; on reaching ``ready`` it refreshes the global
   * ``info`` store so the rest of the UI sees the new capabilities.
   */
  interface Props {
    backend: BackendInfo;
    // Optional: called with the newly-loaded model id whenever a load
    // finishes ``ready`` (and once on mount if the slot is already ready).
    // The Decode page uses this to keep its ``model`` field in sync with
    // whatever the host actually has loaded.
    onReady?: (model: string | null) => void;
    // Compact layout for embedding in the Decode sidebar (no outer border).
    compact?: boolean;
  }
  let { backend, onReady, compact = false }: Props = $props();

  let status = $state<RemoteStatus | null>(null);
  let models = $state<string[]>([]);
  let selected = $state<string>('');
  let modelsNote = $state('');
  let busy = $state(false);
  let error = $state<string | null>(null);
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  // Live elapsed-seconds counter shown next to the progress bar while a
  // load is in flight. Large GGUF models take tens of seconds to mmap +
  // warm up and the host gives us no percentage, so an honest "still
  // working, Ns elapsed" indeterminate bar beats a frozen-looking UI.
  let elapsedSec = $state(0);

  const slotState = $derived<RemoteSlotState>(status?.state ?? 'unknown');

  // Tick the elapsed counter for as long as a load is busy. The effect
  // re-runs when ``busy`` flips; its cleanup clears the interval so we
  // never leak a timer when the load finishes or the component unmounts.
  $effect(() => {
    if (!busy) {
      elapsedSec = 0;
      return;
    }
    const start = Date.now();
    elapsedSec = 0;
    const t = setInterval(() => {
      elapsedSec = Math.floor((Date.now() - start) / 1000);
    }, 250);
    return () => clearInterval(t);
  });

  function labelFor(id: string): string {
    // GGUF ids are absolute paths; show the basename but keep the full id
    // as the value / title. HF ids stay as-is.
    const parts = id.split('/');
    return parts[parts.length - 1] || id;
  }

  function badgeClass(s: RemoteSlotState): string {
    if (s === 'ready') return 'text-emerald-300 bg-emerald-500/10';
    if (s === 'loading') return 'text-sky-300 bg-sky-500/10';
    if (s === 'error') return 'text-rose-300 bg-rose-500/10';
    if (s === 'empty') return 'text-amber-300 bg-amber-500/10';
    return 'text-slate-400 bg-slate-500/10';
  }

  function badgeText(s: RemoteSlotState): string {
    if (s === 'ready') return 'ready';
    if (s === 'loading') return 'loading…';
    if (s === 'error') return 'error';
    if (s === 'empty') return 'no model loaded';
    return 'unknown';
  }

  async function refreshStatus(): Promise<RemoteSlotState> {
    try {
      status = await getRemoteStatus(backend.name);
      error = null;
      // Default the picker to whatever's loaded, once, if untouched.
      if (!selected && status.loaded_model) selected = status.loaded_model;
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    }
    return status?.state ?? 'unknown';
  }

  async function loadModels() {
    try {
      const resp = await apiFetch<ModelsResponse>(
        `/api/v1/models/${encodeURIComponent(backend.name)}`
      );
      models = resp.models ?? [];
      modelsNote = resp.note ?? '';
      if (!selected && models.length > 0) selected = models[0];
    } catch (exc) {
      modelsNote = exc instanceof ApiError ? exc.message : String(exc);
    }
  }

  function stopPolling() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  function schedulePoll() {
    stopPolling();
    pollTimer = setTimeout(async () => {
      const s = await refreshStatus();
      if (s === 'loading') {
        schedulePoll();
      } else {
        busy = false;
        // A finished load (success or failure) changes capabilities /
        // loaded model -- refresh the shared info store so every page's
        // backend dropdown + capability badges update.
        await info.refresh();
        if (s === 'ready') onReady?.(status?.loaded_model ?? null);
      }
    }, 2000);
  }

  async function doReload() {
    busy = true;
    error = null;
    try {
      status = await reloadRemoteModel(backend.name, selected || null);
      if (status.state === 'loading') {
        schedulePoll();
      } else {
        busy = false;
        await info.refresh();
        if (status.state === 'ready') onReady?.(status.loaded_model ?? null);
      }
    } catch (exc) {
      busy = false;
      error = exc instanceof ApiError ? exc.message : String(exc);
    }
  }

  onMount(async () => {
    await Promise.all([refreshStatus(), loadModels()]);
    if (status?.state === 'loading') {
      busy = true;
      schedulePoll();
    } else if (status?.state === 'ready') {
      onReady?.(status.loaded_model ?? null);
    }
  });

  onDestroy(stopPolling);
</script>

<div class="remote-control" class:compact>
  <div class="flex items-center justify-between gap-2 mb-2">
    <div class="flex items-center gap-2">
      <span class="font-mono text-sm text-slate-200">{backend.name}</span>
      <span class="badge {badgeClass(slotState)}">{badgeText(slotState)}</span>
    </div>
    <span class="text-xs text-slate-500 font-mono truncate max-w-[55%]" title={status?.loaded_model || ''}>
      {status?.loaded_model || '—'}
    </span>
  </div>

  {#if slotState === 'error' && status?.error}
    <p class="text-xs text-rose-400 mb-2 break-words">load error: {status.error}</p>
  {/if}

  <div class="flex items-end gap-2">
    <div class="flex-1">
      <label class="label" for={`model-${backend.name}`}>Model on host</label>
      <select
        id={`model-${backend.name}`}
        class="input font-mono text-xs"
        bind:value={selected}
        disabled={busy}
      >
        {#if models.length === 0}
          <option value="">{modelsNote || 'no models found on host'}</option>
        {:else}
          {#each models as m}
            <option value={m} title={m}>{labelFor(m)}</option>
          {/each}
        {/if}
      </select>
    </div>
    <button
      class="btn btn-primary text-sm whitespace-nowrap"
      onclick={doReload}
      disabled={busy || !selected}
      title="(Re)load the selected model on the remote host"
    >
      {busy ? 'loading…' : slotState === 'ready' ? 'Reload' : 'Load'}
    </button>
  </div>

  {#if busy}
    <div class="mt-2" role="status" aria-live="polite">
      <div class="progress-track">
        <div class="progress-bar"></div>
      </div>
      <p class="text-[10px] text-slate-400 mt-1 font-mono">
        loading model… {elapsedSec}s — large models can take a while to
        warm up; the interface is not frozen.
      </p>
    </div>
  {/if}

  {#if modelsNote && models.length > 0}
    <p class="text-[10px] text-slate-500 mt-1 font-mono">{modelsNote}</p>
  {/if}
  {#if error}
    <p class="text-xs text-rose-400 mt-1 break-words">{error}</p>
  {/if}
</div>

<style>
  .remote-control {
    border: 1px solid rgb(51 65 85);
    border-radius: 0.5rem;
    padding: 0.75rem;
    background: rgb(15 23 42 / 0.4);
  }
  .remote-control.compact {
    border: none;
    padding: 0;
    background: transparent;
  }
  .badge {
    font-size: 0.7rem;
    padding: 0.1rem 0.45rem;
    border-radius: 0.375rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  /* Indeterminate progress: the host can't report a load percentage, so
     a sliding bar communicates "work in progress" without lying about
     how far along it is. */
  .progress-track {
    height: 4px;
    width: 100%;
    background: rgb(51 65 85 / 0.6);
    border-radius: 9999px;
    overflow: hidden;
  }
  .progress-bar {
    height: 100%;
    width: 40%;
    border-radius: 9999px;
    background: linear-gradient(
      90deg,
      transparent,
      rgb(56 189 248),
      transparent
    );
    animation: indeterminate 1.2s ease-in-out infinite;
  }
  @keyframes indeterminate {
    0% {
      transform: translateX(-110%);
    }
    100% {
      transform: translateX(310%);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .progress-bar {
      animation-duration: 2.4s;
    }
  }
</style>
