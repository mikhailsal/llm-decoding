<script lang="ts">
  // Phase 5 of "Unify Decode Workbench": the standalone /manual page
  // is gone. Manual is now the rightmost button on the unified
  // /generate (Decode) page; per-pick state lives entirely in the
  // browser and each pick fires a /generate/stream call with
  // prefix_token_ids = picks-so-far. This minimal page exists so any
  // bookmarked /manual link lands on the right tool.
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  onMount(() => {
    goto('/generate', { replaceState: true });
  });
</script>

<div class="text-center text-slate-400 text-sm py-12">
  <p>The standalone <span class="font-mono">/manual</span> page has moved.</p>
  <p class="mt-1">
    Redirecting to the unified <a class="text-sky-400 underline" href="/generate">Decode</a>
    workbench…
  </p>
</div>
