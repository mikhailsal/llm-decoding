<script lang="ts">
  import { confidenceClass, fmtProb, probWidth } from '$lib/render';

  interface Props {
    /** Probability in [0, 1]. */
    prob: number | null | undefined;
    showLabel?: boolean;
  }
  let { prob, showLabel = true }: Props = $props();
</script>

<div class="flex items-center gap-2 w-32">
  <div class="h-2 flex-1 bg-slate-800 rounded overflow-hidden">
    <div
      class="h-full {confidenceClass(prob)} transition-all duration-150 ease-out"
      style="width: {probWidth(prob)}%"
    ></div>
  </div>
  {#if showLabel}
    <span class="text-xs tabular-nums text-slate-300 w-14 text-right">
      {fmtProb(prob)}
    </span>
  {/if}
</div>
