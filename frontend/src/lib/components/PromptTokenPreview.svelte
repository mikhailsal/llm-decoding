<script lang="ts">
  /**
   * Live token preview for the Decode workbench prompt textarea.
   *
   * Why this exists: when the user is on a backend that has a real
   * local tokenizer (HF / llamacpp-py / openai-compat with a mapped
   * HF tokenizer such as Fireworks gpt-oss / Qwen / Llama), we can
   * tokenize their prompt LOCALLY in the web layer and surface the
   * result as a row of chips under the textarea. This turns the
   * sandbox into an honest little teaching tool for "see how your
   * text becomes tokens" -- the single most common question new
   * users ask when they first hit ``Once upon a time`` -> 4 ids and
   * wonder how the model "sees" the text.
   *
   * Mechanics:
   *
   * - Debounce by 250 ms (configurable) so the user can type a
   *   sentence without hammering the web server with one
   *   ``/api/v1/tokenize`` request per keystroke.
   * - ``AbortController`` cancels any in-flight request when a newer
   *   one fires, so the chips never flicker back to a stale state
   *   while a slower request lands second.
   * - When backend + model report ``supports_local_tokenize=false``
   *   the component renders nothing (the fallback synthetic-id stub
   *   would teach the wrong thing -- one chip for the whole prompt).
   * - When the prompt grows beyond ``maxVisibleChips`` we truncate
   *   the chip rendering and show a ``... and N more`` tail, so very
   *   long prompts don't tank the DOM. Token count is always exact.
   * - Errors are surfaced inline ("preview unavailable: HTTP 503")
   *   rather than as toasts: the rest of the workbench keeps working
   *   even when tokenization breaks, and the user sees why.
   */
  import { onDestroy } from 'svelte';
  import TokenInline from './TokenInline.svelte';
  import { isSpecialText } from '$lib/render';
  import { apiFetch, ApiError } from '$lib/api';
  import { info } from '$lib/stores/info';

  interface Props {
    /** Current prompt text (bound to the workbench's textarea). */
    text: string;
    /** Backend identifier the workbench is currently using. */
    backend: string;
    /** Active model id (forwarded to the tokenize endpoint). */
    model: string;
    /**
     * Capability gate. When false the component renders nothing -- the
     * backend either has no real tokenizer (chat-only, no mapping) or
     * its tokenizer failed to load (gated repo without HF_TOKEN).
     */
    enabled: boolean;
    /** ms to wait after the last keystroke before issuing a request. */
    debounceMs?: number;
    /** Show at most this many chips inline; the rest fold into a tail. */
    maxVisibleChips?: number;
  }

  let {
    text,
    backend,
    model,
    enabled,
    debounceMs = 250,
    maxVisibleChips = 200
  }: Props = $props();

  interface TokenizePreview {
    ids: number[];
    pieces: string[];
  }

  let preview = $state<TokenizePreview | null>(null);
  let busy = $state(false);
  let error = $state<string>('');
  let pending: ReturnType<typeof setTimeout> | null = null;
  let abortCtrl: AbortController | null = null;

  // We track the (backend, model, text) tuple that PRODUCED the
  // current preview separately from the props so we never claim a
  // stale preview is current after the user switched backends mid-
  // flight. ``hash`` is a cheap stringification used only for the
  // equality check; collisions are harmless (we'd re-render the same
  // chips).
  let previewKey = $state<string>('');
  function inputKey(): string {
    return `${backend}|${model}|${text}`;
  }
  // Track which backend×model combos we've already refreshed info for
  // after their first successful tokenize. Lets us upgrade stale
  // bos_token_ids (which the static-config path can't supply for cloud
  // backends until the loaded backend has actually pulled the
  // tokenizer.json) without spamming /api/v1/info on every keystroke.
  let infoRefreshedFor = new Set<string>();

  function clear(): void {
    if (pending !== null) {
      clearTimeout(pending);
      pending = null;
    }
    if (abortCtrl) {
      abortCtrl.abort();
      abortCtrl = null;
    }
  }

  async function runTokenize(
    snapshot: string,
    forBackend: string,
    forModel: string
  ): Promise<void> {
    if (!enabled) return;
    if (!forBackend) return;
    if (snapshot === '') {
      preview = { ids: [], pieces: [] };
      previewKey = `${forBackend}|${forModel}|${snapshot}`;
      error = '';
      return;
    }
    clear();
    busy = true;
    error = '';
    const ctrl = new AbortController();
    abortCtrl = ctrl;
    try {
      const data = await apiFetch<TokenizePreview>('/api/v1/tokenize', {
        method: 'POST',
        body: JSON.stringify({ backend: forBackend, model: forModel, text: snapshot }),
        signal: ctrl.signal
      });
      // Only adopt the response if it still corresponds to the LATEST
      // input. If the user kept typing or switched backends while we
      // were in flight, drop the stale data and let the next debounced
      // call land instead -- prevents a brief "wrong chips, then
      // right chips" flicker.
      if (snapshot !== text || forBackend !== backend || forModel !== model) {
        return;
      }
      preview = {
        ids: data.ids,
        pieces: (data.pieces && data.pieces.length === data.ids.length)
          ? data.pieces
          : []
      };
      previewKey = `${forBackend}|${forModel}|${snapshot}`;
      // First successful tokenize for this backend×model means the
      // upstream is now loaded and its capabilities (bos_token_ids
      // in particular) are accurate. Trigger ONE info refresh so the
      // workbench's "fill BOS" button picks up the freshly-discovered
      // ids without waiting for a full page reload. Subsequent
      // tokenize calls on the same combo are no-ops here.
      const refreshKey = `${forBackend}|${forModel}`;
      if (!infoRefreshedFor.has(refreshKey)) {
        infoRefreshedFor.add(refreshKey);
        void info.refresh();
      }
    } catch (err) {
      // AbortError fires on every "user typed another char while we
      // were mid-flight" path. That's not an error condition for the
      // user -- just suppress and let the next call land.
      if ((err as Error).name === 'AbortError') return;
      if (err instanceof ApiError) {
        error = `preview unavailable: HTTP ${err.status}`;
      } else {
        error = `preview unavailable: ${(err as Error).message || 'unknown'}`;
      }
    } finally {
      if (abortCtrl === ctrl) abortCtrl = null;
      busy = false;
    }
  }

  // Debounced retrigger on (text / backend / model / enabled) change.
  // ``$effect`` is the idiomatic Svelte 5 way to react to props.
  $effect(() => {
    // Touch every reactive input the debounce should depend on. The
    // ``snapshot`` is captured at the moment the user pauses typing,
    // so the in-flight request always matches what the user saw the
    // moment we decided to fire.
    const snapshot = text;
    const _b = backend;
    const _m = model;
    const _e = enabled;
    if (!enabled) {
      clear();
      preview = null;
      error = '';
      return;
    }
    if (pending !== null) clearTimeout(pending);
    const forBackend = backend;
    const forModel = model;
    pending = setTimeout(() => {
      pending = null;
      void runTokenize(snapshot, forBackend, forModel);
    }, debounceMs);
  });

  onDestroy(() => clear());

  // Derived view: chips + truncation. Empty when preview is null.
  interface Chip {
    id: number;
    piece: string;
    isSpecial: boolean;
  }
  let chips = $derived<Chip[]>(
    preview
      ? preview.ids.map((id, i) => {
          const piece = preview!.pieces[i] ?? '';
          return {
            id,
            piece,
            isSpecial: isSpecialText(piece)
          };
        })
      : []
  );
  let truncated = $derived(chips.length > maxVisibleChips);
  let shownChips = $derived(truncated ? chips.slice(0, maxVisibleChips) : chips);
  let isFresh = $derived(previewKey === inputKey());
</script>

{#if enabled}
  <div class="tok-preview" data-testid="prompt-token-preview">
    <div class="tok-preview-header">
      <span
        class="label"
        title="Tokens shown here are produced LOCALLY using the same tokenizer the provider runs server-side ({model}). Use this to learn how your text becomes tokens before generation. Click a chip to copy its id."
      >
        token preview
        {#if preview}
          · <span class="tok-count">{chips.length} {chips.length === 1 ? 'token' : 'tokens'}</span>
        {/if}
        {#if busy}
          · <span class="tok-busy">…</span>
        {/if}
        {#if !isFresh && preview && !busy}
          · <span class="tok-stale" title="The preview is for an older prompt; new tokens are being computed">stale</span>
        {/if}
      </span>
      {#if error}
        <span class="tok-err" title={error}>{error}</span>
      {/if}
    </div>
    {#if preview === null && !busy && !error}
      <div class="tok-empty">type to see how your prompt becomes tokens…</div>
    {:else if preview && chips.length === 0}
      <div class="tok-empty">(empty prompt)</div>
    {:else}
      <div class="tok-chips">
        {#each shownChips as chip, i (i)}
          <span class="tok-chip" title={`id: ${chip.id}`}>
            <TokenInline
              text={chip.piece || ''}
              isSpecial={chip.isSpecial}
              showMarkers={true}
            />
            <span class="tok-id">{chip.id}</span>
          </span>
        {/each}
        {#if truncated}
          <span class="tok-chip-more" title="{chips.length - maxVisibleChips} more tokens not rendered for performance; counts are still exact">
            … +{chips.length - maxVisibleChips} more
          </span>
        {/if}
      </div>
    {/if}
  </div>
{/if}

<style>
  .tok-preview {
    margin-top: 0.25rem;
    font-size: 0.78rem;
    line-height: 1.4;
  }
  .tok-preview-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.5rem;
    margin-bottom: 0.2rem;
    color: var(--color-text-muted, #666);
  }
  .tok-count {
    font-variant-numeric: tabular-nums;
    color: var(--color-text-strong, #222);
  }
  .tok-busy {
    color: #888;
  }
  .tok-stale {
    color: #b58000;
    font-style: italic;
  }
  .tok-err {
    color: #c33;
  }
  .tok-empty {
    color: #999;
    font-style: italic;
    padding: 0.2rem 0;
  }
  .tok-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.2rem 0.25rem;
    padding: 0.25rem 0.3rem;
    background: rgba(0, 0, 0, 0.02);
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 0.25rem;
    max-height: 12rem;
    overflow-y: auto;
  }
  .tok-chip {
    display: inline-flex;
    align-items: baseline;
    gap: 0.25rem;
    padding: 0.05rem 0.35rem;
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: 0.25rem;
    line-height: 1.3;
  }
  .tok-id {
    font-size: 0.65rem;
    color: #888;
    font-variant-numeric: tabular-nums;
  }
  .tok-chip-more {
    display: inline-flex;
    align-items: baseline;
    padding: 0.05rem 0.35rem;
    color: #888;
    font-style: italic;
  }
  :global(.dark) .tok-chips {
    background: rgba(255, 255, 255, 0.03);
    border-color: rgba(255, 255, 255, 0.08);
  }
  :global(.dark) .tok-chip {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(255, 255, 255, 0.1);
  }
  :global(.dark) .tok-id {
    color: #aaa;
  }
</style>
