<script lang="ts">
  import { renderTokenSegments, type TokenSegment } from '$lib/render';

  /**
   * Render one decoded token inline inside a running-completion paragraph,
   * with an optional background tint by per-token probability.
   *
   * Why a separate component from ``TokenText``? Two reasons:
   *
   * - The running-completion view sometimes wants to *hide* the ``␣ ↵ →``
   *   visibility markers so the text reads like normal prose. ``TokenText``
   *   always shows them (it's used in the per-row table where you need to
   *   distinguish ``" I"`` from ``"I "`` at a glance).
   * - The running-completion view colors each token by probability with a
   *   semi-transparent background; ``TokenText`` is wrapped in table cells
   *   where the background is fixed by the row.
   */
  interface Props {
    text: string;
    isSpecial?: boolean;
    showMarkers?: boolean;
    bgClass?: string;
    title?: string;
  }
  let {
    text,
    isSpecial = false,
    showMarkers = true,
    bgClass = '',
    title = ''
  }: Props = $props();

  let segments = $derived<TokenSegment[]>(
    showMarkers
      ? renderTokenSegments(text, isSpecial)
      : isSpecial
        ? [{ kind: 'special', text: text || '<special>' }]
        : text === ''
          ? [{ kind: 'empty', text: '<empty>' }]
          : [{ kind: 'plain', text }]
  );
</script>

<span
  class={`token-inline ${bgClass}`}
  title={title}
  style="white-space: pre-wrap;"
>
  {#each segments as seg}
    {#if seg.kind === 'plain'}<span>{seg.text}</span
      >{:else if seg.kind === 'ws'}<span class="tok-ws">{seg.text}</span
      >{:else if seg.kind === 'special'}<span class="tok-special">{seg.text}</span
      >{:else if seg.kind === 'control'}<span class="tok-ws">{seg.text}</span
      >{:else}<span class="tok-empty">{seg.text}</span>{/if}
  {/each}
</span>

<style>
  .token-inline {
    border-radius: 0.25rem;
    padding: 0 1px;
    line-height: 1.7;
  }
</style>
