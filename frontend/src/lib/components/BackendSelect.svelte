<script lang="ts">
  import { info } from '$lib/stores/info';

  interface Props {
    value: string;
    onChange?: (value: string) => void;
    /** When true, only backends with `available=true` are selectable. */
    onlyAvailable?: boolean;
    label?: string;
    id?: string;
  }
  let { value = $bindable(), onChange, onlyAvailable = false, label = 'Backend', id = 'backend' }: Props = $props();

  function handle(e: Event) {
    const next = (e.target as HTMLSelectElement).value;
    value = next;
    onChange?.(next);
  }
</script>

<div>
  <label class="label" for={id}>{label}</label>
  <select {id} class="input font-mono" value={value} onchange={handle}>
    {#if $info.info}
      {#each $info.info.backends as b}
        {@const inert = !!b.capabilities?.generation_disabled}
        {@const inertNote = inert ? b.capabilities?.notes || 'generation disabled' : ''}
        {@const optDisabled = (onlyAvailable && !b.available) || inert}
        {@const optTitle = inert ? inertNote : b.note || ''}
        {@const suffix = !b.available
          ? ` — ${b.note || 'unavailable'}`
          : inert
            ? ` — ${inertNote}`
            : ''}
        <option value={b.name} disabled={optDisabled} title={optTitle}>
          {b.label}{suffix}
        </option>
      {/each}
    {/if}
  </select>
</div>
