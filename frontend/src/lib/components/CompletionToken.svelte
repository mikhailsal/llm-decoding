<script lang="ts">
  import { renderTokenSegments, formatProbPct, rankCandidates, type TokenSegment } from '$lib/render';
  import type { TokenCandidate } from '$lib/types';

  /**
   * Interactive token used in the running-completion view AND in the
   * generation/prompt tables. It unifies what used to be two divergent
   * affordances (the read-only ``TokenInline`` in the completion and the
   * logit-bias-only ``biasable`` snippet in the tables, the latter
   * clickable only on backends that support ``logit_bias`` — which is why
   * dsbx-host-py tokens were inert while fireworks tokens weren't).
   *
   * Behaviour:
   *   - HOVER (or keyboard focus): a popover shows the top-K alternative
   *     tokens with their probabilities AND the action buttons together.
   *     Each action appears only when its callback is provided, so the
   *     same component serves the completion (find / prompt / watch) and
   *     the tables (prompt / watch / bias-when-supported).
   *   - CLICK: runs the primary action -- "find in list" (jump to this
   *     token's row in the steps table below) when available. The other
   *     actions live in the hover popover, so a plain click is a fast path
   *     to the most common "where did this come from" lookup.
   *
   * WHITESPACE INVARIANT: ``.ct-root`` inherits ``white-space: pre-wrap``
   * from the running-completion container, so ANY source whitespace
   * between the inner token ``<span>`` and the ``{#if popoverOpen}`` block
   * (or between ``{/if}`` and the root ``</span>``) renders as a literal,
   * preserved space AFTER every token -- a visible gap between adjacent
   * tokens. Keep those inline siblings strictly flush (no newlines).
   */
  interface Props {
    text: string;
    isSpecial?: boolean;
    tokenId?: number | null;
    showMarkers?: boolean;
    bgClass?: string;
    className?: string;
    /** Top-K alternatives shown on hover (optional). */
    candidates?: TokenCandidate[] | null;
    /** Native-title fallback (e.g. the chosen-token probability). */
    probTitle?: string;
    onWatch?: (id: number, text: string) => void;
    onPrompt?: (text: string) => void;
    onFind?: () => void;
    onBias?: (id: number) => void;
  }
  let {
    text,
    isSpecial = false,
    tokenId = null,
    showMarkers = true,
    bgClass = '',
    className = '',
    candidates = null,
    probTitle = '',
    onWatch,
    onPrompt,
    onFind,
    onBias
  }: Props = $props();

  let rootEl: HTMLElement;
  let hovering = $state(false);
  let focused = $state(false);

  // A real, biasable/watchable token id (finite, not a synthetic intern
  // id >= 1<<24 that the upstream never sees).
  const realId = $derived<number | null>(
    tokenId !== null && tokenId !== undefined && Number.isFinite(tokenId) && tokenId < 1 << 24
      ? tokenId
      : null
  );

  const segments = $derived<TokenSegment[]>(
    showMarkers
      ? renderTokenSegments(text, isSpecial)
      : isSpecial
        ? [{ kind: 'special', text: text || '<special>' }]
        : text === ''
          ? [{ kind: 'empty', text: '<empty>' }]
          : [{ kind: 'plain', text }]
  );

  // Sort by raw logprob (chosen-first on ties) so the selected token is
  // always the top row -- see rankCandidates. Show the position-based rank
  // (loop index) rather than the backend ``rank`` so the numbering matches
  // this re-sorted order.
  const topAlts = $derived<TokenCandidate[]>(
    rankCandidates(candidates, realId).slice(0, 8)
  );
  const hasAlts = $derived<boolean>(topAlts.length > 0);
  const hasActions = $derived<boolean>(
    !!onPrompt || (!!onWatch && realId !== null) || !!onFind || (!!onBias && realId !== null)
  );

  // The popover (alternatives + actions) opens on hover OR keyboard focus
  // -- no click needed. It stays open while the pointer is anywhere in the
  // root subtree (the popover is a DOM descendant, so moving onto it does
  // not fire the root's mouseleave), which is how the action buttons stay
  // reachable.
  const popoverOpen = $derived<boolean>(
    (hovering || focused) && (hasAlts || hasActions)
  );

  /** Primary action for a plain click: jump to this token in the steps
   *  list. No-op for table cells (which have no list to find). */
  function primary() {
    if (onFind) onFind();
  }

  function doWatch() {
    if (onWatch && realId !== null) onWatch(realId, text);
  }
  function doPrompt() {
    if (onPrompt) onPrompt(text);
  }
  function doFind() {
    if (onFind) onFind();
  }
  function doBias() {
    if (onBias && realId !== null) onBias(realId);
  }

  function segmentClass(kind: TokenSegment['kind']): string {
    if (kind === 'ws' || kind === 'control') return 'tok-ws';
    if (kind === 'special') return 'tok-special';
    if (kind === 'empty') return 'tok-empty';
    return '';
  }
</script>

<span
  bind:this={rootEl}
  class="ct-root"
  onmouseenter={() => (hovering = true)}
  onmouseleave={() => (hovering = false)}
  onfocusin={() => (focused = true)}
  onfocusout={() => (focused = false)}
  role="group"
>
  <span
    class={`token-inline ${bgClass} ${className}`}
    style="white-space: pre-wrap;"
    role="button"
    tabindex="0"
    class:ct-interactive={hasActions || hasAlts}
    onclick={primary}
    onkeydown={(e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        primary();
      }
    }}
    title={probTitle}
  >{#each segments as seg}<span class={segmentClass(seg.kind)}>{seg.text}</span>{/each}</span>{#if popoverOpen}<span class="ct-popover">
      {#if hasAlts}
        <span class="ct-section-label">alternatives</span>
        <span class="ct-alts">
          {#each topAlts as c, ci}
            <span class="ct-alt" class:ct-alt-chosen={realId !== null && c.token_id === realId}>
              <span class="ct-alt-rank">{ci + 1}</span>
              <span class="ct-alt-text font-mono">{c.text === '' ? '<empty>' : c.text}</span>
              <span class="ct-alt-prob">{formatProbPct(c.logprob)}</span>
            </span>
          {/each}
        </span>
      {/if}

      {#if hasActions}
        <span class="ct-actions">
          {#if onFind}
            <button type="button" class="ct-action" onclick={doFind}>find in list</button>
          {/if}
          {#if onPrompt}
            <button type="button" class="ct-action" onclick={doPrompt}>add to prompt</button>
          {/if}
          {#if onWatch && realId !== null}
            <button type="button" class="ct-action" onclick={doWatch}>add to watch</button>
          {/if}
          {#if onBias && realId !== null}
            <button type="button" class="ct-action" onclick={doBias}>add to logit bias</button>
          {/if}
        </span>
      {/if}
    </span>{/if}</span>

<style>
  .ct-root {
    position: relative;
    border-radius: 0.25rem;
    cursor: default;
  }
  .ct-interactive {
    cursor: pointer;
  }
  .ct-interactive:hover {
    outline: 1px solid rgb(56 189 248 / 0.5);
    outline-offset: 0;
  }
  .token-inline {
    border-radius: 0.25rem;
    padding: 0 1px;
    line-height: 1.7;
  }
  .ct-popover {
    position: absolute;
    top: 100%;
    left: 0;
    z-index: 60;
    /* Flush against the token's bottom edge: a hover-only popover must
       leave no dead gap, or the pointer crossing the gap would fire the
       root's mouseleave and close it before reaching the buttons. */
    margin-top: 0;
    min-width: 12rem;
    max-width: 20rem;
    display: block;
    background: rgb(15 23 42);
    border: 1px solid rgb(51 65 85);
    border-radius: 0.5rem;
    padding: 0.4rem;
    box-shadow: 0 8px 24px rgb(0 0 0 / 0.5);
    white-space: normal;
    cursor: default;
  }
  .ct-section-label {
    display: block;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: rgb(100 116 139);
    margin-bottom: 0.2rem;
  }
  .ct-alts {
    display: block;
  }
  .ct-alt {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    padding: 0.05rem 0.15rem;
    border-radius: 0.25rem;
  }
  .ct-alt-chosen {
    background: rgb(56 189 248 / 0.15);
  }
  .ct-alt-rank {
    color: rgb(100 116 139);
    width: 1rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .ct-alt-text {
    flex: 1;
    color: rgb(226 232 240);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: pre;
  }
  .ct-alt-prob {
    color: rgb(148 163 184);
    font-variant-numeric: tabular-nums;
  }
  .ct-actions {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    margin-top: 0.35rem;
    padding-top: 0.35rem;
    border-top: 1px solid rgb(51 65 85 / 0.6);
  }
  .ct-action {
    text-align: left;
    font-size: 0.72rem;
    color: rgb(186 230 253);
    padding: 0.2rem 0.35rem;
    border-radius: 0.25rem;
    background: transparent;
  }
  .ct-action:hover {
    background: rgb(56 189 248 / 0.15);
  }
  :global(.ct-root .tok-ws) {
    color: rgb(100 116 139);
  }
  :global(.ct-root .tok-special) {
    color: rgb(217 70 239);
  }
  :global(.ct-root .tok-empty) {
    color: rgb(100 116 139);
    font-style: italic;
  }
</style>
