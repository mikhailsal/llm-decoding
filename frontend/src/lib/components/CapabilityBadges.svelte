<script lang="ts">
  import { info } from '$lib/stores/info';
  import type { Capabilities } from '$lib/types';

  interface Props {
    backend: string | null;
  }
  let { backend }: Props = $props();

  let caps = $derived<Capabilities | null>(
    backend && $info.info
      ? $info.info.backends.find((b) => b.name === backend)?.capabilities ?? null
      : null
  );
</script>

{#if caps}
  <div class="flex flex-wrap gap-1.5">
    {#if caps.full_vocab}
      <span class="chip" title="next_distribution returns the full vocab">full-vocab</span>
    {/if}
    {#if caps.prompt_logprobs}
      <span class="chip" title="score_prompt returns logprobs for every prompt token">prompt-logprobs</span>
    {/if}
    {#if caps.can_force_token}
      <span class="chip" title="manual force-by-text / force-by-id is allowed">force-token</span>
    {/if}
    <span class="chip" title="max value of top_k">top_k≤{caps.max_top_logprobs}</span>
    {#if caps.eos_token_ids?.length}
      <span class="chip" title={`EOS token ids: ${caps.eos_token_ids.join(', ')}`}>
        EOS:{caps.eos_token_ids.length}
      </span>
    {/if}
  </div>
{/if}
