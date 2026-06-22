<script lang="ts">
  // Phase 5 of "Unify Decode Workbench": the standalone /inspect page
  // is gone. Inspect is now the leftmost button on the unified
  // /generate (Decode) page; behind the scenes it fires the same
  // /generate/stream call with max_tokens=1 + include_prompt=true.
  // This minimal page exists so any bookmarked /inspect link lands on
  // the right tool (client-side replace, no flicker on the static
  // adapter bundle).
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  onMount(() => {
    goto('/generate', { replaceState: true });
  });
</script>

<div class="text-center text-slate-400 text-sm py-12">
  <p>The standalone <span class="font-mono">/inspect</span> page has moved.</p>
  <p class="mt-1">
    Redirecting to the unified <a class="text-sky-400 underline" href="/generate">Decode</a>
    workbench…
  </p>
</div>
