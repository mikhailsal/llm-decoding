<script lang="ts">
  import type { BackendInfo } from '$lib/types';

  /**
   * Model picker that adapts its UI to whatever the selected backend can do.
   *
   * Three rendering modes, picked from ``BackendInfo``:
   *
   * 1. ``model_editable = true`` and a non-empty ``suggested_models`` list:
   *    render a ``<select>`` over the suggestions plus a "custom..."
   *    option that swaps to a text input. This is the cloud-provider path
   *    where the OpenAI-compatible backend accepts any model name.
   * 2. ``model_editable = true`` and no suggestions: render a plain text
   *    input (the user types the model id).
   * 3. ``model_editable = false``: render a disabled, read-only field
   *    that shows ``loaded_model`` (or "unknown" until the upstream
   *    has been contacted). This is the remote / local-engine path
   *    where switching models means restarting a heavy process.
   *
   * The component is fully ``$bindable`` -- the parent owns the string.
   * Switching backends from the parent should reset ``value`` to the new
   * backend's ``loaded_model`` (or empty) explicitly; the component doesn't
   * silently mutate the binding.
   */
  interface Props {
    backend: BackendInfo | null;
    value: string;
    onChange?: (value: string) => void;
    id?: string;
    label?: string;
  }
  let {
    backend,
    value = $bindable(),
    onChange,
    id = 'model',
    label = 'Model'
  }: Props = $props();

  let editable = $derived(!!backend?.model_editable);
  let suggestions = $derived(backend?.suggested_models ?? []);
  let useSelect = $derived(editable && suggestions.length > 0);
  // "custom..." is the always-trailing escape hatch for editable backends.
  const CUSTOM = '__custom__';
  let pickMode = $state<'preset' | 'custom'>('preset');

  // If the parent flips ``value`` to something not in the suggestions, we
  // automatically switch to custom mode so the input shows the right thing.
  $effect(() => {
    if (!editable) return;
    if (!value) {
      pickMode = 'preset';
      return;
    }
    if (suggestions.includes(value)) {
      pickMode = 'preset';
    } else {
      pickMode = 'custom';
    }
  });

  function handleSelect(e: Event) {
    const next = (e.target as HTMLSelectElement).value;
    if (next === CUSTOM) {
      pickMode = 'custom';
      // Don't clobber an in-progress custom value.
      if (!value || suggestions.includes(value)) {
        value = '';
        onChange?.('');
      }
      return;
    }
    pickMode = 'preset';
    value = next;
    onChange?.(next);
  }

  function handleInput(e: Event) {
    const next = (e.target as HTMLInputElement).value;
    value = next;
    onChange?.(next);
  }
</script>

<div>
  <label class="label" for={id}>{label}</label>
  {#if !editable}
    <div
      class="input font-mono cursor-not-allowed bg-slate-900/60 text-slate-400"
      id={id}
      title={backend?.note || ''}
    >
      {backend?.loaded_model || 'unknown until backend is contacted'}
    </div>
    {#if backend?.family === 'remote'}
      <p class="text-xs text-slate-500 mt-1">
        Loaded by ``dsbx serve`` on the remote host; restart it with a different
        ``--model`` to switch.
      </p>
    {/if}
  {:else if useSelect}
    <div class="flex gap-2">
      <select
        id={id}
        class="input font-mono flex-1"
        value={pickMode === 'custom' ? CUSTOM : value || ''}
        onchange={handleSelect}
      >
        {#each suggestions as m}
          <option value={m}>{m}</option>
        {/each}
        <option value={CUSTOM}>custom…</option>
      </select>
    </div>
    {#if pickMode === 'custom'}
      <input
        type="text"
        class="input font-mono mt-2"
        placeholder="enter provider-specific model id"
        value={value || ''}
        oninput={handleInput}
      />
    {/if}
  {:else}
    <input
      type="text"
      class="input font-mono"
      id={id}
      placeholder="model id"
      value={value || ''}
      oninput={handleInput}
    />
  {/if}
</div>
