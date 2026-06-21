<script lang="ts">
  import { renderTokenSegments, type TokenSegment } from '$lib/render';

  interface Props {
    text: string;
    isSpecial?: boolean;
    /** Optional extra Tailwind classes (e.g. "font-mono text-sm"). */
    className?: string;
  }
  let { text, isSpecial = false, className = 'font-mono' }: Props = $props();

  let segments = $derived<TokenSegment[]>(renderTokenSegments(text, isSpecial));
</script>

<span class={className}>
  {#each segments as seg}
    {#if seg.kind === 'plain'}
      <span>{seg.text}</span>
    {:else if seg.kind === 'ws'}
      <span class="tok-ws">{seg.text}</span>
    {:else if seg.kind === 'special'}
      <span class="tok-special">{seg.text}</span>
    {:else if seg.kind === 'control'}
      <span class="tok-ws">{seg.text}</span>
    {:else}
      <span class="tok-empty">{seg.text}</span>
    {/if}
  {/each}
</span>
