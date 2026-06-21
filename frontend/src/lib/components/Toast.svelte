<script lang="ts">
  interface Props {
    message: string | null;
    kind?: 'error' | 'info' | 'success';
    onClose?: () => void;
  }
  let { message, kind = 'error', onClose }: Props = $props();

  const tone = $derived(
    kind === 'error'
      ? 'bg-rose-500/15 border-rose-500/40 text-rose-200'
      : kind === 'success'
      ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-200'
      : 'bg-sky-500/15 border-sky-500/40 text-sky-200'
  );
</script>

{#if message}
  <div class="fixed top-4 right-4 z-50 max-w-md">
    <div class="border rounded-md px-3 py-2 text-sm shadow-lg flex items-start gap-2 {tone}">
      <span class="flex-1 font-mono">{message}</span>
      {#if onClose}
        <button class="text-current opacity-80 hover:opacity-100" onclick={onClose}>×</button>
      {/if}
    </div>
  </div>
{/if}
