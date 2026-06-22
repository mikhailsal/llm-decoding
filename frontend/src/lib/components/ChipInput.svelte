<script lang="ts">
  interface Props {
    values: string[];
    label: string;
    placeholder?: string;
    /** When true, leading/trailing space is preserved -- needed for --watch ' Paris'. */
    preserveSpace?: boolean;
    hint?: string;
  }
  let { values = $bindable(), label, placeholder = 'enter, press space', preserveSpace = true, hint = '' }: Props = $props();
  let pending = $state('');

  function add() {
    const v = preserveSpace ? pending : pending.trim();
    if (!v) return;
    values = [...values, v];
    pending = '';
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      add();
    } else if (e.key === 'Backspace' && pending === '' && values.length) {
      values = values.slice(0, -1);
    }
  }

  // Commit pending text whenever the input loses focus. Without this,
  // a user who typed ``who`` into the Stop-text field and then clicked
  // ``generate`` would silently lose that entry because the chip was
  // never committed (Enter wasn't pressed). Click events on buttons
  // fire AFTER blur on the previously-focused input, so the parent's
  // ``run()`` handler sees the freshly committed ``values``.
  function onBlur() {
    if (pending) add();
  }

  function remove(i: number) {
    values = values.filter((_, k) => k !== i);
  }

  function fmt(v: string): string {
    // Show leading/trailing spaces with ␣ so the user can verify what
    // they actually typed (matches the TUI's --watch ' Paris' convention).
    return v
      .replace(/^ +/, (m) => '\u2423'.repeat(m.length))
      .replace(/ +$/, (m) => '\u2423'.repeat(m.length));
  }
</script>

<div>
  <label class="label">{label}</label>
  <div class="flex flex-wrap gap-1.5 mb-1">
    {#each values as v, i}
      <span class="chip">
        <span class="font-mono">{fmt(v)}</span>
        <button class="text-slate-400 hover:text-rose-400" onclick={() => remove(i)} aria-label="remove">
          ×
        </button>
      </span>
    {/each}
  </div>
  <input
    type="text"
    class="input font-mono"
    bind:value={pending}
    onkeydown={onKey}
    onblur={onBlur}
    placeholder={placeholder}
  />
  {#if hint}
    <p class="text-xs text-slate-500 mt-1">{hint}</p>
  {/if}
</div>
