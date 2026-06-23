<script lang="ts">
  /**
   * Token-aware prompt composer for the Decode workbench.
   *
   * Replaces the old "plain textarea on the left + separate token preview
   * on the right + separate prepend-ids chip-input" trio with a single
   * editable field that:
   *
   *   1. Highlights token boundaries INLINE, as a background-coloured
   *      backdrop directly behind the text (no numbers, same spirit as the
   *      running-completion colouring) -- so the student sees how their
   *      text splits into tokens AS THEY TYPE.
   *   2. Offers a model-specific palette of EVERY special token the
   *      tokenizer knows (BOS / EOS / chat / tool markers), fetched from
   *      ``/api/v1/special_tokens``. Clicking one splices its exact string
   *      at the caret; because the backend tokenizes with special-token
   *      matching on, that string round-trips to the single control-token
   *      id it names. This is how "condition on BOS" now works: just insert
   *      the BOS token at the start of the prompt -- no separate prepend
   *      field needed.
   *   3. Lets advanced users insert an arbitrary token BY ID: we resolve
   *      the id to its surface text via ``/api/v1/piece`` and splice that.
   *      (Plain non-special pieces are re-tokenized on the server, so this
   *      is "usually exact" rather than guaranteed -- special tokens, which
   *      DO round-trip, are the precise path.)
   *
   * The highlight backdrop ALWAYS renders text sliced from the live ``value``
   * (never from the pieces directly), so the overlay stays pixel-aligned with
   * the textarea even when a tokenizer's per-piece decode doesn't perfectly
   * reconstruct the source string.
   */
  import { onDestroy, tick } from 'svelte';
  import { apiFetch, ApiError } from '$lib/api';
  import { isSpecialText } from '$lib/render';

  interface Props {
    /** Bound prompt text (the model input being composed). */
    value: string;
    backend: string;
    model: string;
    /**
     * When false the backend has no real local tokenizer: we fall back to
     * a plain textarea (no highlight, no palette) so the field still works
     * but never lies with a one-chip-for-everything stub.
     */
    enabled: boolean;
    placeholder?: string;
    rows?: number;
    debounceMs?: number;
  }

  let {
    value = $bindable(),
    backend,
    model,
    enabled,
    placeholder = 'Type your prompt. Insert special tokens from the palette below.',
    rows = 6,
    debounceMs = 200
  }: Props = $props();

  // ---- live tokenization (debounced + abortable) ---------------------- //
  interface TokenizePreview {
    ids: number[];
    pieces: string[];
  }
  let preview = $state<TokenizePreview | null>(null);
  let busy = $state(false);
  let tokError = $state('');
  let previewKey = $state('');
  let pending: ReturnType<typeof setTimeout> | null = null;
  let abortCtrl: AbortController | null = null;

  function inputKey(): string {
    return `${backend}|${model}|${value}`;
  }

  function clearPending(): void {
    if (pending !== null) {
      clearTimeout(pending);
      pending = null;
    }
    if (abortCtrl) {
      abortCtrl.abort();
      abortCtrl = null;
    }
  }

  async function runTokenize(snapshot: string, b: string, m: string): Promise<void> {
    if (!enabled || !b) return;
    if (snapshot === '') {
      preview = { ids: [], pieces: [] };
      previewKey = `${b}|${m}|${snapshot}`;
      tokError = '';
      return;
    }
    clearPending();
    busy = true;
    tokError = '';
    const ctrl = new AbortController();
    abortCtrl = ctrl;
    try {
      const data = await apiFetch<TokenizePreview>('/api/v1/tokenize', {
        method: 'POST',
        body: JSON.stringify({ backend: b, model: m, text: snapshot }),
        signal: ctrl.signal
      });
      if (snapshot !== value || b !== backend || m !== model) return;
      preview = {
        ids: data.ids,
        pieces:
          data.pieces && data.pieces.length === data.ids.length ? data.pieces : []
      };
      previewKey = `${b}|${m}|${snapshot}`;
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      tokError =
        err instanceof ApiError
          ? `tokenize failed: HTTP ${err.status}`
          : `tokenize failed: ${(err as Error).message || 'unknown'}`;
    } finally {
      if (abortCtrl === ctrl) abortCtrl = null;
      busy = false;
    }
  }

  $effect(() => {
    const snapshot = value;
    const b = backend;
    const m = model;
    const e = enabled;
    if (!e) {
      clearPending();
      preview = null;
      tokError = '';
      return;
    }
    if (pending !== null) clearTimeout(pending);
    pending = setTimeout(() => {
      pending = null;
      void runTokenize(snapshot, b, m);
    }, debounceMs);
  });

  onDestroy(() => clearPending());

  // ---- backdrop segmentation (always covers the exact value) ---------- //
  let isFresh = $derived(previewKey === inputKey());
  let tokenCount = $derived(preview ? preview.ids.length : 0);

  interface Seg {
    text: string;
    special: boolean;
    idx: number;
  }
  // Greedily walk the source string, attributing each piece's length to a
  // slice OF THE SOURCE (so concatenated segments === value, guaranteeing
  // alignment). When a piece matches at the cursor we advance by its
  // length; if a tokenizer's decode diverges from the source we still
  // advance by the piece length so we never desync the whole field.
  let segments = $derived.by<Seg[]>(() => {
    if (!preview || !isFresh || preview.pieces.length === 0) return [];
    const text = value;
    const pieces = preview.pieces;
    const segs: Seg[] = [];
    let pos = 0;
    for (let k = 0; k < pieces.length && pos <= text.length; k++) {
      const pc = pieces[k] ?? '';
      const len = pc.length;
      if (len <= 0) continue;
      const take = Math.min(len, text.length - pos);
      if (take <= 0) break;
      segs.push({ text: text.slice(pos, pos + take), special: isSpecialText(pc), idx: k });
      pos += take;
    }
    if (pos < text.length) {
      segs.push({ text: text.slice(pos), special: false, idx: pieces.length });
    }
    return segs;
  });

  // ---- special-token palette ----------------------------------------- //
  interface SpecialTok {
    id: number;
    text: string;
  }
  let specials = $state<SpecialTok[]>([]);
  let specialsErr = $state('');
  let specialsKey = $state('');
  let paletteOpen = $state(true);
  let search = $state('');
  let showNoise = $state(false);
  const MAX_PALETTE = 80;

  // Some tokenizers pad their special vocab with hundreds/thousands of
  // inert markers -- DeepSeek ships 800 ``<｜place▁holder▁no▁N｜>`` + 415
  // ``<|place_holder_mm_span_N|>`` (1215 of 1230!), gpt-oss has
  // ``<|reserved_NNN|>``. These bury the ~15 genuinely useful tokens past
  // the visible cap. We bucket them as "noise" and hide them by default
  // (a toggle reveals them; search always covers the FULL set so nothing
  // is permanently unreachable).
  function isNoiseSpecial(text: string): boolean {
    const norm = text.replace(/[<>|\uFF5C\u2581_ .]/g, '').toLowerCase();
    return /(placeholder|reserved|unused)/.test(norm);
  }

  $effect(() => {
    const b = backend;
    const m = model;
    if (!enabled || !b) {
      specials = [];
      return;
    }
    const key = `${b}|${m}`;
    if (key === specialsKey) return;
    specialsKey = key;
    specialsErr = '';
    void apiFetch<{ tokens: SpecialTok[] }>('/api/v1/special_tokens', {
      method: 'POST',
      body: JSON.stringify({ backend: b, model: m })
    })
      .then((r) => {
        if (`${backend}|${model}` !== key) return;
        specials = r.tokens ?? [];
      })
      .catch((err) => {
        if (`${backend}|${model}` !== key) return;
        specials = [];
        specialsErr =
          err instanceof ApiError ? `HTTP ${err.status}` : (err as Error).message;
      });
  });

  let usefulSpecials = $derived<SpecialTok[]>(
    specials.filter((s) => !isNoiseSpecial(s.text))
  );
  let noiseCount = $derived(specials.length - usefulSpecials.length);

  let filteredSpecials = $derived.by<SpecialTok[]>(() => {
    const q = search.trim().toLowerCase();
    // While searching, search the FULL set (incl. noise) so a known
    // reserved/placeholder id is still reachable; otherwise show useful
    // only unless the user opted into the noise via the toggle.
    const base = q ? specials : showNoise ? specials : usefulSpecials;
    if (!q) return base;
    return base.filter(
      (s) => s.text.toLowerCase().includes(q) || String(s.id).includes(q)
    );
  });
  let shownSpecials = $derived(filteredSpecials.slice(0, MAX_PALETTE));

  // ---- insert by id --------------------------------------------------- //
  let idInput = $state('');
  let idErr = $state('');

  async function insertById(): Promise<void> {
    idErr = '';
    const id = Number.parseInt(idInput, 10);
    if (!Number.isFinite(id)) {
      idErr = 'enter a numeric id';
      return;
    }
    try {
      const r = await apiFetch<{ text: string }>('/api/v1/piece', {
        method: 'POST',
        body: JSON.stringify({ backend, model, id })
      });
      if (!r.text) {
        idErr = `id ${id} has no surface text`;
        return;
      }
      insertAtCaret(r.text);
      idInput = '';
    } catch (err) {
      idErr = err instanceof ApiError ? `HTTP ${err.status}` : (err as Error).message;
    }
  }

  // ---- caret-aware insertion ----------------------------------------- //
  let taEl = $state<HTMLTextAreaElement | null>(null);
  let backdropEl = $state<HTMLDivElement | null>(null);

  function insertAtCaret(s: string): void {
    const el = taEl;
    const start = el ? el.selectionStart : value.length;
    const end = el ? el.selectionEnd : value.length;
    value = value.slice(0, start) + s + value.slice(end);
    void tick().then(() => {
      if (!el) return;
      const pos = start + s.length;
      el.focus();
      el.setSelectionRange(pos, pos);
      syncScroll();
    });
  }

  function syncScroll(): void {
    if (taEl && backdropEl) {
      backdropEl.scrollTop = taEl.scrollTop;
      backdropEl.scrollLeft = taEl.scrollLeft;
    }
  }
</script>

<div class="composer">
  <div class="composer-head">
    <span class="label-inline">prompt</span>
    {#if enabled}
      <span class="meta">
        {#if busy}<span class="dot">…</span>{/if}
        <span class="count">{tokenCount} {tokenCount === 1 ? 'token' : 'tokens'}</span>
        {#if !isFresh && preview && !busy}
          <span class="stale" title="recomputing tokens for the latest text">stale</span>
        {/if}
        {#if tokError}<span class="err" title={tokError}>{tokError}</span>{/if}
      </span>
    {/if}
  </div>

  {#if enabled}
    <div class="editor">
      <div class="backdrop" bind:this={backdropEl} aria-hidden="true">{#if segments.length}{#each segments as seg (seg.idx)}<span
              class="tok {seg.special ? 'tok-special' : seg.idx % 2 === 0 ? 'tok-a' : 'tok-b'}"
              >{seg.text}</span
            >{/each}{:else}<span class="tok-plain">{value}</span>{/if}{#if value.endsWith('\n')}<span>&nbsp;</span>{/if}</div>
      <textarea
        bind:this={taEl}
        bind:value
        {rows}
        {placeholder}
        spellcheck="false"
        class="ta"
        onscroll={syncScroll}
      ></textarea>
    </div>
  {:else}
    <textarea bind:value {rows} placeholder="prompt (no local tokenizer on this backend)" class="ta ta-plain"></textarea>
  {/if}

  {#if enabled}
    <div class="palette">
      <button
        type="button"
        class="palette-toggle"
        onclick={() => (paletteOpen = !paletteOpen)}
        aria-expanded={paletteOpen}
      >
        {paletteOpen ? '▾' : '▸'} special tokens
        <span class="palette-count">{usefulSpecials.length}</span>
        {#if specialsErr}<span class="err">({specialsErr})</span>{/if}
      </button>
      {#if paletteOpen && noiseCount > 0}
        <button
          type="button"
          class="noise-toggle"
          onclick={() => (showNoise = !showNoise)}
          title="Reserved / placeholder / unused markers the tokenizer pads its vocab with. They're hidden by default because they bury the useful tokens; search always finds them regardless."
        >
          {showNoise ? 'hide' : 'show'} {noiseCount} reserved/placeholder
        </button>
      {/if}

      {#if paletteOpen}
        {#if specials.length === 0}
          <p class="palette-empty">
            {specialsErr
              ? 'could not load special tokens for this model'
              : 'this model exposes no special tokens'}
          </p>
        {:else}
          <div class="palette-controls">
            <input
              type="text"
              class="palette-search"
              placeholder="filter by name or id…"
              bind:value={search}
            />
            <div class="by-id">
              <input
                type="text"
                inputmode="numeric"
                class="id-input"
                placeholder="id"
                bind:value={idInput}
                onkeydown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    void insertById();
                  }
                }}
              />
              <button type="button" class="id-btn" onclick={() => void insertById()}>
                insert by id
              </button>
            </div>
          </div>
          {#if idErr}<p class="err id-err">{idErr}</p>{/if}
          <div class="chips">
            {#each shownSpecials as s (s.id)}
              <button
                type="button"
                class="chip"
                title={`insert ${s.text} (id ${s.id})`}
                onclick={() => insertAtCaret(s.text)}
              >
                <span class="chip-text">{s.text}</span>
                <span class="chip-id">{s.id}</span>
              </button>
            {/each}
            {#if filteredSpecials.length > shownSpecials.length}
              <span class="more">
                +{filteredSpecials.length - shownSpecials.length} more — refine the filter
              </span>
            {/if}
          </div>
        {/if}
        <p class="hint">
          Inserting a special token splices its exact string at the cursor; it
          tokenizes back to that single id. Put the model's BOS at the very start
          to condition generation on it (replaces the old “prepend BOS”).
        </p>
      {/if}
    </div>
  {/if}
</div>

<style>
  .composer {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .composer-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.72rem;
  }
  .label-inline {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
  }
  .meta {
    display: inline-flex;
    gap: 0.45rem;
    align-items: baseline;
    color: #94a3b8;
  }
  .count {
    font-variant-numeric: tabular-nums;
    color: #cbd5e1;
  }
  .stale {
    color: #b58000;
    font-style: italic;
  }
  .dot {
    color: #64748b;
  }
  .err {
    color: #f87171;
  }

  /* The overlay: backdrop + textarea share identical box metrics so the
     coloured token spans sit exactly behind the glyphs. */
  .editor {
    position: relative;
  }
  .backdrop,
  .ta {
    margin: 0;
    border: 1px solid rgb(51 65 85);
    border-radius: 0.375rem;
    padding: 0.5rem 0.625rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.85rem;
    line-height: 1.5;
    white-space: pre-wrap;
    overflow-wrap: break-word;
    word-break: break-word;
    box-sizing: border-box;
    width: 100%;
    min-height: calc(6 * 1.5em);
  }
  .backdrop {
    position: absolute;
    inset: 0;
    overflow: auto;
    pointer-events: none;
    color: transparent;
    background: rgb(15 23 42);
    z-index: 0;
  }
  .ta {
    position: relative;
    z-index: 1;
    background: transparent;
    color: rgb(226 232 240);
    caret-color: rgb(56 189 248);
    resize: vertical;
  }
  .ta::placeholder {
    color: #64748b;
  }
  .ta-plain {
    background: rgb(15 23 42);
  }
  .ta:focus,
  .ta-plain:focus {
    outline: none;
    border-color: rgb(56 189 248);
  }
  /* Token backgrounds: pure background only (no padding/margin) so the
     overlay never shifts a single glyph. Alternating shades make boundaries
     legible; specials pop in sky. */
  .tok {
    border-radius: 2px;
  }
  .tok-a {
    background: rgba(148, 163, 184, 0.1);
  }
  .tok-b {
    background: rgba(148, 163, 184, 0.22);
  }
  .tok-special {
    background: rgba(217, 70, 239, 0.35);
    color: transparent;
    box-shadow: 0 0 0 1px rgba(217, 70, 239, 0.55) inset;
  }
  .tok-plain {
    color: transparent;
  }

  /* palette */
  .palette {
    font-size: 0.75rem;
  }
  .palette-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: #cbd5e1;
    background: none;
    border: none;
    cursor: pointer;
    padding: 0.1rem 0;
  }
  .palette-count {
    font-variant-numeric: tabular-nums;
    color: #64748b;
    background: rgba(148, 163, 184, 0.15);
    border-radius: 999px;
    padding: 0 0.4rem;
  }
  .noise-toggle {
    margin-left: 0.5rem;
    background: none;
    border: none;
    color: #64748b;
    cursor: pointer;
    text-decoration: underline dotted;
    font-size: 0.72rem;
    padding: 0;
  }
  .noise-toggle:hover {
    color: #94a3b8;
  }
  .palette-empty {
    color: #64748b;
    font-style: italic;
    margin: 0.25rem 0;
  }
  .palette-controls {
    display: flex;
    gap: 0.4rem;
    margin: 0.35rem 0;
    flex-wrap: wrap;
  }
  .palette-search,
  .id-input {
    background: rgb(15 23 42);
    border: 1px solid rgb(51 65 85);
    border-radius: 0.375rem;
    padding: 0.25rem 0.45rem;
    color: rgb(226 232 240);
    font-size: 0.75rem;
  }
  .palette-search {
    flex: 1 1 12rem;
    min-width: 8rem;
  }
  .id-input {
    width: 5rem;
    font-family: ui-monospace, monospace;
  }
  .by-id {
    display: inline-flex;
    gap: 0.3rem;
  }
  .id-btn,
  .chip {
    background: rgba(148, 163, 184, 0.12);
    border: 1px solid rgb(51 65 85);
    border-radius: 0.375rem;
    color: #cbd5e1;
    cursor: pointer;
  }
  .id-btn {
    padding: 0.25rem 0.55rem;
    font-size: 0.72rem;
    white-space: nowrap;
  }
  .id-btn:hover {
    border-color: rgb(100 116 139);
  }
  .id-err {
    margin: 0.15rem 0;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    max-height: 11rem;
    overflow-y: auto;
    padding: 0.15rem 0.1rem;
  }
  .chip {
    display: inline-flex;
    align-items: baseline;
    gap: 0.3rem;
    padding: 0.1rem 0.4rem;
    font-family: ui-monospace, monospace;
    font-size: 0.72rem;
  }
  .chip:hover {
    border-color: rgba(217, 70, 239, 0.6);
    background: rgba(217, 70, 239, 0.15);
  }
  .chip-text {
    color: #e9d5ff;
  }
  .chip-id {
    color: #64748b;
    font-size: 0.62rem;
    font-variant-numeric: tabular-nums;
  }
  .more {
    color: #64748b;
    font-style: italic;
    align-self: center;
  }
  .hint {
    color: #64748b;
    line-height: 1.4;
    margin: 0.35rem 0 0;
  }
</style>
