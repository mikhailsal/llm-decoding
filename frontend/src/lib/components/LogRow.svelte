<!--
  One row of the upstream-request log list. Renders the at-a-glance fields
  (timestamp, backend, method+path, status, latency, token totals,
  truncated completion preview) with light colour-coding for status and
  latency. The /logs page wraps this in a clickable container that opens
  the detail panel on the right.
-->
<script lang="ts">
  import type { LogSummary } from '$lib/types';

  interface Props {
    log: LogSummary;
    selected?: boolean;
    onSelect?: (id: string) => void;
  }

  let { log, selected = false, onSelect }: Props = $props();

  function statusClass(code: number | null, err: string | null): string {
    if (err) return 'text-rose-400';
    if (code === null) return 'text-slate-500';
    if (code >= 500) return 'text-rose-400';
    if (code >= 400) return 'text-amber-300';
    if (code >= 200) return 'text-emerald-300';
    return 'text-slate-300';
  }

  function statusLabel(code: number | null, err: string | null): string {
    if (err && code === null) return 'ERR';
    if (code === null) return '—';
    return String(code);
  }

  function latencyClass(ms: number | null): string {
    if (ms === null) return 'text-slate-500';
    if (ms < 500) return 'text-emerald-300';
    if (ms < 2000) return 'text-sky-300';
    if (ms < 10000) return 'text-amber-300';
    return 'text-rose-400';
  }

  function formatLatency(ms: number | null): string {
    if (ms === null) return '—';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  function formatTime(iso: string): string {
    try {
      const d = new Date(iso);
      // HH:mm:ss.SSS in local time -- precise enough to tell back-to-back
      // requests apart without monopolizing the row.
      return d.toLocaleTimeString(undefined, { hour12: false }) +
        '.' + String(d.getMilliseconds()).padStart(3, '0');
    } catch {
      return iso;
    }
  }

  function tokensLabel(prompt: number | null, completion: number | null): string {
    if (prompt === null && completion === null) return '—';
    const p = prompt ?? 0;
    const c = completion ?? 0;
    return `${p}→${c}`;
  }
</script>

<button
  class="w-full text-left grid grid-cols-[7rem_8rem_1fr_4rem_5rem_6rem] gap-2 px-3 py-2
         border-b border-slate-800 hover:bg-slate-800/40 transition-colors
         {selected ? 'bg-sky-500/10 border-l-2 border-l-sky-400' : ''}"
  type="button"
  onclick={() => onSelect?.(log.id)}
>
  <span class="font-mono text-xs text-slate-400 truncate" title={log.timestamp}>
    {formatTime(log.timestamp)}
  </span>
  <span class="text-xs truncate" title="{log.backend_name} ({log.backend_family})">
    <span class="text-slate-300 font-medium">{log.backend_name}</span>
    {#if log.is_streaming}
      <span class="text-violet-300 text-[10px] uppercase ml-1">stream</span>
    {/if}
  </span>
  <span class="text-xs truncate" title={log.upstream_path}>
    <span class="font-mono text-slate-500">{log.method}</span>
    <span class="font-mono text-slate-300 ml-1">{log.upstream_path || '/'}</span>
    {#if log.completion_text}
      <span class="text-slate-500 ml-2 italic">·</span>
      <span class="text-slate-400 ml-1">{log.completion_text}</span>
    {:else if log.error_message}
      <span class="text-rose-400 ml-2">{log.error_message}</span>
    {/if}
  </span>
  <span class="text-xs font-mono {statusClass(log.response_status_code, log.error_message)}">
    {statusLabel(log.response_status_code, log.error_message)}
  </span>
  <span class="text-xs font-mono {latencyClass(log.latency_ms)}" title={
    log.ttft_ms !== null ? `TTFT: ${Math.round(log.ttft_ms)}ms` : ''
  }>
    {formatLatency(log.latency_ms)}
  </span>
  <span class="text-xs font-mono text-slate-400" title="prompt → completion tokens">
    {tokensLabel(log.prompt_tokens, log.completion_tokens)}
  </span>
</button>
