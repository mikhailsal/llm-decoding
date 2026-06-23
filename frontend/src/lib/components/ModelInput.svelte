<script lang="ts">
  import type { BackendInfo, ModelsResponse } from '$lib/types';
  import { apiFetch, ApiError } from '$lib/api';

  /**
   * Model picker that adapts its UI to whatever the selected backend can do.
   *
   * Two render modes, derived from ``BackendInfo``:
   *
   * 1. ``model_editable = true``: a filterable combobox -- an ``<input>``
   *    whose value is the chosen model id, plus a dropdown of suggestions
   *    that the user can substring-filter as they type. On first focus
   *    we lazily ask ``/api/v1/models/{name}`` for the live catalogue
   *    (cached on the middleware for 6h); a refresh button forces a
   *    re-fetch by setting ``?refresh=true``. This is the only mode the
   *    Fireworks/NIM/OpenRouter pickers ever use.
   * 2. ``model_editable = false``: a read-only field showing
   *    ``loaded_model`` exactly -- never truncated visually, so the
   *    full GGUF path stays clickable/copyable; it just wraps to extra
   *    lines for very long values. This is the dsbx-host-py path; switching
   *    the loaded model means restarting ``dsbx serve`` and the
   *    middleware will not silently change it.
   *
   * The ``value`` prop is ``$bindable``: the parent owns the string. A
   * parent that switches the backend is expected to reset ``value`` to
   * the new backend's ``loaded_model``; the component never mutates the
   * binding on its own except in response to user input.
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
  let staticSuggestions = $derived(backend?.suggested_models ?? []);

  let liveModels = $state<string[] | null>(null);
  let liveSource = $state<ModelsResponse['source'] | null>(null);
  let liveFetchedAt = $state<number | null>(null);
  let liveNote = $state('');
  let loading = $state(false);
  let loadError = $state<string | null>(null);

  // Combobox state: ``open`` controls dropdown visibility; ``query`` is the
  // current filter text; ``activeIdx`` is the keyboard-focused row.
  let open = $state(false);
  let query = $state('');
  let activeIdx = $state(0);
  let inputEl = $state<HTMLInputElement | null>(null);
  let listEl = $state<HTMLDivElement | null>(null);
  // ``editing`` tracks whether the user is typing into the box vs the
  // input is mirroring the bound ``value``. When false the input shows
  // ``value`` verbatim; when true it shows the in-progress ``query``.
  let editing = $state(false);

  let allOptions = $derived<string[]>(
    liveModels && liveModels.length > 0 ? liveModels : staticSuggestions
  );

  let filtered = $derived<string[]>(
    query.trim() === ''
      ? allOptions
      : allOptions.filter((m) =>
          m.toLowerCase().includes(query.trim().toLowerCase())
        )
  );

  // Reset filter every time we open the dropdown so the user sees the
  // whole list first; only narrow when they actually type.
  function openDropdown() {
    if (!editable) return;
    open = true;
    activeIdx = 0;
    editing = false;
    query = '';
    if (liveModels === null && !loading) {
      void loadModels(false);
    }
  }

  function closeDropdown() {
    open = false;
    editing = false;
    query = '';
  }

  async function loadModels(forceRefresh: boolean) {
    if (!backend) return;
    loading = true;
    loadError = null;
    try {
      const q = forceRefresh ? '?refresh=true' : '';
      const resp = await apiFetch<ModelsResponse>(
        `/api/v1/models/${encodeURIComponent(backend.name)}${q}`
      );
      liveModels = resp.models ?? [];
      liveSource = resp.source;
      liveFetchedAt = resp.fetched_at;
      liveNote = resp.note ?? '';
    } catch (exc) {
      loadError =
        exc instanceof ApiError ? exc.message : exc instanceof Error ? exc.message : String(exc);
    } finally {
      loading = false;
    }
  }

  function commit(next: string) {
    value = next;
    onChange?.(next);
    closeDropdown();
  }

  function handleInputFocus() {
    openDropdown();
  }

  function handleInput(e: Event) {
    const input = e.target as HTMLInputElement;
    editing = true;
    query = input.value;
    activeIdx = 0;
    open = true;
    // Free-text values are always valid -- the cloud backend accepts any
    // model id, so we propagate the typed text immediately.
    value = input.value;
    onChange?.(input.value);
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (!editable) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!open) {
        openDropdown();
        return;
      }
      activeIdx = Math.min(filtered.length - 1, activeIdx + 1);
      scrollActiveIntoView();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (!open) return;
      activeIdx = Math.max(0, activeIdx - 1);
      scrollActiveIntoView();
    } else if (e.key === 'Enter') {
      if (open && filtered.length > 0) {
        e.preventDefault();
        commit(filtered[activeIdx] ?? value);
      } else {
        closeDropdown();
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      closeDropdown();
    }
  }

  function scrollActiveIntoView() {
    if (!listEl) return;
    const row = listEl.querySelector<HTMLElement>(`[data-idx="${activeIdx}"]`);
    row?.scrollIntoView({ block: 'nearest' });
  }

  function handleBlur(e: FocusEvent) {
    // Don't close if the focus is moving into the dropdown (e.g. clicking
    // a row). The row's mousedown handler will commit and close.
    const related = e.relatedTarget as HTMLElement | null;
    if (related && listEl?.contains(related)) return;
    closeDropdown();
  }

  function refresh() {
    void loadModels(true);
  }

  // The trailing button is a DROPDOWN TOGGLE (chevron), not a refresh
  // icon: a bare ``↻`` next to the field read as "three dots that do
  // nothing" because clicking it refetched the catalogue invisibly while
  // the list was closed. Now it opens/closes the list like any combobox;
  // an explicit "refresh" link lives inside the open list for the rare
  // re-fetch case.
  function toggleDropdown() {
    if (!editable) return;
    if (open) {
      closeDropdown();
    } else {
      openDropdown();
      inputEl?.focus();
    }
  }

  function fmtSource(): string {
    if (!liveSource) return '';
    if (liveSource === 'live') return 'live';
    if (liveSource === 'cached') {
      if (liveFetchedAt == null) return 'cached';
      const ageS = Math.max(0, Math.floor(Date.now() / 1000 - liveFetchedAt));
      if (ageS < 60) return `cached (${ageS}s ago)`;
      if (ageS < 3600) return `cached (${Math.floor(ageS / 60)}m ago)`;
      return `cached (${Math.floor(ageS / 3600)}h ago)`;
    }
    if (liveSource === 'fallback') return 'fallback (curated only)';
    return 'static';
  }
</script>

<div class="model-input">
  <label class="label" for={id}>{label}</label>
  {#if !editable}
    <div
      class="input font-mono cursor-not-allowed bg-slate-900/60 text-slate-400 model-readonly"
      id={id}
      title={backend?.loaded_model || ''}
    >
      {backend?.loaded_model || 'unknown until backend is contacted'}
    </div>
    {#if backend?.family === 'remote'}
      <p class="text-xs text-slate-500 mt-1">
        Loaded by <span class="font-mono">dsbx serve</span> on the remote host. Switch it from the
        <a href="/status" class="text-sky-400 hover:underline">Status</a> page's remote model control.
      </p>
    {/if}
  {:else}
    <div class="combobox">
      <div class="combobox-input-row">
        <input
          bind:this={inputEl}
          type="text"
          class="input font-mono flex-1"
          id={id}
          placeholder="type to filter… or enter any provider-specific id"
          value={editing ? query : value || ''}
          onfocus={handleInputFocus}
          oninput={handleInput}
          onkeydown={handleKeyDown}
          onblur={handleBlur}
          autocomplete="off"
          spellcheck="false"
        />
        <button
          type="button"
          class="btn btn-ghost text-xs px-2 combobox-toggle"
          class:open
          onclick={toggleDropdown}
          aria-expanded={open}
          aria-label="show model list"
          title={open ? 'hide model list' : 'show model list'}
        >
          {loading ? '…' : '▾'}
        </button>
      </div>
      {#if open}
        <div
          bind:this={listEl}
          class="combobox-list"
          role="listbox"
          tabindex={-1}
        >
          {#if loading && filtered.length === 0}
            <div class="combobox-empty">loading catalogue…</div>
          {:else if loadError}
            <div class="combobox-empty text-rose-400">{loadError}</div>
          {:else if filtered.length === 0}
            <div class="combobox-empty">no matches</div>
          {:else}
            {#each filtered as m, i}
              <button
                type="button"
                class="combobox-row"
                class:active={i === activeIdx}
                class:selected={m === value}
                data-idx={i}
                onmousedown={(e) => {
                  e.preventDefault();
                  commit(m);
                }}
                onmouseenter={() => (activeIdx = i)}
              >
                {m}
              </button>
            {/each}
          {/if}
          <div class="combobox-footer">
            <button
              type="button"
              class="combobox-refresh"
              onmousedown={(e) => {
                e.preventDefault();
                refresh();
              }}
              disabled={loading}
              title="re-fetch the provider's model catalogue from the API"
            >
              {loading ? 'refreshing…' : '↻ refresh catalogue'}
            </button>
          </div>
        </div>
      {/if}
      <p class="combobox-status">
        {#if loading}
          <span class="text-slate-500">loading…</span>
        {:else if liveSource}
          <span class="text-slate-500">
            {filtered.length} of {allOptions.length} ·
            <span class="font-mono">{fmtSource()}</span>{#if liveNote} · {liveNote}{/if}
          </span>
        {:else if allOptions.length > 0}
          <span class="text-slate-500">{allOptions.length} curated suggestions — click ↻ to fetch live</span>
        {/if}
      </p>
    </div>
  {/if}
</div>

<style>
  .model-input {
    position: relative;
  }
  .model-readonly {
    /* Long paths (think ~/.cache/dsbx/.../*.gguf) must remain
       fully readable: wrap rather than truncate, with the full string in
       the title attribute for hover. ``break-all`` is what kicks in when
       there's no whitespace to break at. */
    white-space: normal;
    word-break: break-all;
    line-height: 1.4;
    min-height: 2.25rem;
    height: auto;
  }
  .combobox {
    position: relative;
  }
  .combobox-input-row {
    display: flex;
    gap: 0.375rem;
    align-items: stretch;
  }
  .combobox-toggle {
    transition: transform 0.12s ease;
  }
  .combobox-toggle.open {
    transform: rotate(180deg);
  }
  .combobox-footer {
    border-top: 1px solid rgb(51 65 85);
    margin-top: 0.25rem;
    padding-top: 0.25rem;
    display: flex;
    justify-content: flex-end;
  }
  .combobox-refresh {
    background: none;
    border: 0;
    color: rgb(148 163 184);
    font-size: 0.72rem;
    cursor: pointer;
    padding: 0.2rem 0.35rem;
    border-radius: 0.25rem;
  }
  .combobox-refresh:hover {
    color: rgb(226 232 240);
    background: rgb(30 41 59);
  }
  .combobox-list {
    position: absolute;
    top: calc(100% + 0.25rem);
    left: 0;
    right: 0;
    z-index: 30;
    max-height: 16rem;
    overflow-y: auto;
    background: rgb(15 23 42 / 0.98);
    border: 1px solid rgb(51 65 85);
    border-radius: 0.375rem;
    box-shadow: 0 10px 30px rgb(0 0 0 / 0.4);
    padding: 0.25rem;
  }
  .combobox-row {
    display: block;
    width: 100%;
    text-align: left;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.875rem;
    line-height: 1.4;
    padding: 0.3rem 0.5rem;
    border-radius: 0.25rem;
    background: transparent;
    color: rgb(226 232 240);
    border: 0;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .combobox-row.active {
    background: rgb(30 41 59);
    color: rgb(248 250 252);
  }
  .combobox-row.selected {
    color: rgb(125 211 252);
  }
  .combobox-empty {
    padding: 0.5rem;
    font-size: 0.8rem;
    color: rgb(148 163 184);
  }
  .combobox-status {
    margin-top: 0.25rem;
    font-size: 0.7rem;
    line-height: 1.2;
  }
</style>
