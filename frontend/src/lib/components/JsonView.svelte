<!--
  Recursive collapsible JSON renderer used by the /logs detail panel.

  Built on a native <details> element so the chevron is the system one and
  the keyboard accessibility (Tab + Space) comes for free. The component is
  self-recursive: for object / array values it renders another <JsonView />
  inside the open <details>, so deeply nested structures collapse the same
  way as the top level.

  Why not a JSON.stringify() pre? Because the operator wants to drill into
  one branch of a 100k+ token completion response without scrolling past
  the rest. Native <details> + flexbox gives that for free.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import Self from './JsonView.svelte';

  interface Props {
    data: unknown;
    name?: string | null;
    depth?: number;
    /** Open the top-level node automatically. */
    open?: boolean;
  }

  let { data, name = null, depth = 0, open = true }: Props = $props();

  let kind = $derived(getKind(data));
  let asArray = $derived(kind === 'array' ? (data as unknown[]) : null);
  let asObject = $derived(kind === 'object' ? (data as Record<string, unknown>) : null);
  let objectKeys = $derived(asObject ? Object.keys(asObject) : []);

  function getKind(value: unknown): 'object' | 'array' | 'primitive' | 'null' {
    if (value === null || value === undefined) return 'null';
    if (Array.isArray(value)) return 'array';
    if (typeof value === 'object') return 'object';
    return 'primitive';
  }

  function summary(value: unknown): string {
    if (value === null || value === undefined) return 'null';
    if (Array.isArray(value)) return `Array(${value.length})`;
    if (typeof value === 'object') {
      const keys = Object.keys(value as object);
      const preview = keys.slice(0, 3).join(', ');
      const more = keys.length > 3 ? ', …' : '';
      return `{ ${preview}${more} } · ${keys.length} key${keys.length === 1 ? '' : 's'}`;
    }
    if (typeof value === 'string') {
      const trimmed = value.length > 80 ? value.slice(0, 80) + '…' : value;
      return JSON.stringify(trimmed);
    }
    return String(value);
  }

  function primitiveClass(value: unknown): string {
    if (value === null || value === undefined) return 'text-slate-500';
    if (typeof value === 'string') return 'text-emerald-300';
    if (typeof value === 'number') return 'text-amber-300';
    if (typeof value === 'boolean') return 'text-sky-300';
    return 'text-slate-200';
  }

  function primitiveDisplay(value: unknown): string {
    if (value === null || value === undefined) return 'null';
    if (typeof value === 'string') return JSON.stringify(value);
    return String(value);
  }

  // Auto-collapse very large nodes by default so the detail panel doesn't
  // unfold a thousand-element array on first paint.
  let initialOpen = $derived.by(() => {
    if (!open) return false;
    if (asArray && asArray.length > 100) return false;
    if (asObject && Object.keys(asObject).length > 50) return false;
    return depth < 3;
  });

  // Used only on the root render so deep recursion doesn't replay the
  // mount handler.
  let rootEl: HTMLDivElement | null = $state(null);
  onMount(() => {
    // no-op; reserved for future "copy whole tree" affordance
  });
</script>

<!--
  Cases:
  - object / array: <details> with summary + recursive body
  - primitive: span coloured by type
-->
{#if kind === 'object' || kind === 'array'}
  <details class="json-node" open={initialOpen}>
    <summary class="json-summary">
      {#if name !== null}
        <span class="json-key">{name}</span>
        <span class="text-slate-500">:</span>
      {/if}
      <span class="text-slate-400 font-mono text-xs">{summary(data)}</span>
    </summary>
    <div class="json-body" bind:this={rootEl}>
      {#if asArray}
        {#each asArray as item, i}
          <Self data={item} name={String(i)} depth={depth + 1} open={open} />
        {/each}
      {:else if asObject}
        {#each objectKeys as key (key)}
          <Self data={asObject[key]} name={key} depth={depth + 1} open={open} />
        {/each}
      {/if}
    </div>
  </details>
{:else}
  <div class="json-leaf">
    {#if name !== null}
      <span class="json-key">{name}</span>
      <span class="text-slate-500">:</span>
    {/if}
    <span class={primitiveClass(data)}>{primitiveDisplay(data)}</span>
  </div>
{/if}

<style>
  :global(.json-node) {
    padding-left: 1rem;
    border-left: 1px solid rgb(30 41 59);
  }
  :global(.json-summary) {
    cursor: pointer;
    padding: 1px 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
  }
  :global(.json-summary:hover) {
    background-color: rgba(30, 41, 59, 0.5);
  }
  :global(.json-body) {
    padding-left: 0.25rem;
  }
  :global(.json-key) {
    color: rgb(148 163 184);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
  }
  :global(.json-leaf) {
    padding-left: 1rem;
    padding-top: 1px;
    padding-bottom: 1px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
    white-space: pre-wrap;
    word-break: break-word;
  }
</style>
