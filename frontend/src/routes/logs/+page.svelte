<!--
  /logs -- browse upstream HTTP calls captured by LoggingTransport.

  Layout mirrors a production proxy gateway's RequestBrowser/RequestDetail but trimmed for
  our single-side capture (we log only the call we MADE, not the call
  the browser made to us): left pane is a scrollable row list with cursor
  pagination, right pane is a collapsible-JSON detail view of the selected
  row. The header row carries lightweight stats + filters.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import Toast from '$lib/components/Toast.svelte';
  import JsonView from '$lib/components/JsonView.svelte';
  import LogRow from '$lib/components/LogRow.svelte';
  import {
    ApiError,
    deleteLogs,
    getLog,
    getLogStats,
    listLogs,
    searchLogs
  } from '$lib/api';
  import { info } from '$lib/stores/info';
  import type {
    LogDetail,
    LogListResponse,
    LogStats,
    LogSummary
  } from '$lib/types';

  let items = $state<LogSummary[]>([]);
  let nextCursor = $state<string | null>(null);
  let hasMore = $state(false);
  let listBusy = $state(false);
  let stats = $state<LogStats | null>(null);
  let selectedId = $state<string | null>(null);
  let selectedDetail = $state<LogDetail | null>(null);
  let detailBusy = $state(false);
  let error = $state<string | null>(null);

  let filterBackend = $state<string>('');
  let filterError = $state(false);
  let searchQuery = $state('');
  let autoRefresh = $state(false);
  let autoRefreshTimer: ReturnType<typeof setInterval> | null = $state(null);

  let backendOptions = $derived(
    $info.info?.backends.map((b) => b.name) ?? []
  );

  async function refresh(resetSelection = false) {
    listBusy = true;
    try {
      const page = (await listLogs({
        limit: 100,
        backend: filterBackend || null,
        is_error: filterError ? true : null
      })) as LogListResponse;
      items = page.items;
      nextCursor = page.next_cursor;
      hasMore = page.has_more;
      stats = await getLogStats();
      if (resetSelection) {
        selectedId = items.length > 0 ? items[0].id : null;
        if (selectedId) {
          await openDetail(selectedId);
        } else {
          selectedDetail = null;
        }
      }
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    } finally {
      listBusy = false;
    }
  }

  async function loadMore() {
    if (!hasMore || !nextCursor || listBusy) return;
    listBusy = true;
    try {
      const page = (await listLogs({
        cursor: nextCursor,
        limit: 100,
        backend: filterBackend || null,
        is_error: filterError ? true : null
      })) as LogListResponse;
      items = [...items, ...page.items];
      nextCursor = page.next_cursor;
      hasMore = page.has_more;
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    } finally {
      listBusy = false;
    }
  }

  async function openDetail(id: string) {
    selectedId = id;
    detailBusy = true;
    try {
      selectedDetail = await getLog(id);
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
      selectedDetail = null;
    } finally {
      detailBusy = false;
    }
  }

  async function runSearch() {
    const q = searchQuery.trim();
    if (!q) {
      await refresh(true);
      return;
    }
    listBusy = true;
    try {
      const page = (await searchLogs(q, 200)) as LogListResponse;
      items = page.items;
      nextCursor = null;
      hasMore = false;
      selectedId = items.length > 0 ? items[0].id : null;
      if (selectedId) await openDetail(selectedId);
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    } finally {
      listBusy = false;
    }
  }

  async function clearAll() {
    if (!confirm('Delete every log row? This cannot be undone.')) return;
    try {
      await deleteLogs({ all: true });
      await refresh(true);
    } catch (exc) {
      error = exc instanceof ApiError ? exc.message : String(exc);
    }
  }

  function toggleAutoRefresh() {
    autoRefresh = !autoRefresh;
    if (autoRefreshTimer !== null) {
      clearInterval(autoRefreshTimer);
      autoRefreshTimer = null;
    }
    if (autoRefresh) {
      autoRefreshTimer = setInterval(() => {
        if (!listBusy && !searchQuery.trim()) refresh(false);
      }, 5000);
    }
  }

  function formatMs(ms: number | null | undefined): string {
    if (ms === null || ms === undefined) return '—';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  onMount(() => {
    // Kick off the async refresh but return synchronously so onMount can
    // hand us a cleanup callback (Svelte 5 disallows async onMount that
    // returns a cleanup; we don't actually need the cleanup to run after
    // the async work completes).
    (async () => {
      if (!$info.info) {
        try {
          await info.refresh();
        } catch {
          // ignore -- a missing info shouldn't block the logs view
        }
      }
      await refresh(true);
    })();
    return () => {
      if (autoRefreshTimer !== null) clearInterval(autoRefreshTimer);
    };
  });
</script>

<Toast message={error} onClose={() => (error = null)} />

<div class="flex flex-col gap-3">
  <!-- header / stats / filters -->
  <div class="card flex flex-wrap items-center gap-3">
    <h2 class="text-lg font-semibold">Upstream requests</h2>
    {#if stats}
      <span class="chip">total: {stats.total}</span>
      <span class="chip" title="Streaming responses are merged into one row each.">
        streaming: {stats.streaming}
      </span>
      <span class="chip">non-streaming: {stats.non_streaming}</span>
      <span class={`chip ${stats.error_count > 0 ? 'border-rose-500/40 text-rose-300' : ''}`}>
        errors: {stats.error_count}
      </span>
      <span class="chip" title="prompt → completion tokens summed">
        tokens: {stats.total_prompt_tokens}→{stats.total_completion_tokens}
      </span>
      <span class="chip" title="Average end-to-end latency across all rows">
        avg-latency: {formatMs(stats.avg_latency_ms)}
      </span>
      <span class="chip" title="Average time-to-first-token (streaming rows only)">
        avg-ttft: {formatMs(stats.avg_ttft_ms)}
      </span>
    {/if}
    <div class="flex-1"></div>
    <button class="btn btn-ghost text-xs" onclick={() => refresh(false)} disabled={listBusy}>
      {listBusy ? 'loading…' : 'refresh'}
    </button>
    <button
      class={`btn text-xs ${autoRefresh ? 'btn-primary' : 'btn-ghost'}`}
      onclick={toggleAutoRefresh}
      title="Poll every 5s"
    >
      auto · {autoRefresh ? 'on' : 'off'}
    </button>
    <button class="btn btn-ghost text-xs" onclick={clearAll}>
      delete all
    </button>
  </div>

  <div class="card flex flex-wrap items-center gap-3">
    <label class="flex items-center gap-2">
      <span class="text-xs text-slate-400">backend</span>
      <select class="input text-xs py-1 px-2 w-auto" bind:value={filterBackend}
              onchange={() => refresh(true)}>
        <option value="">(all)</option>
        {#each backendOptions as name}
          <option value={name}>{name}</option>
        {/each}
      </select>
    </label>
    <label class="flex items-center gap-2 text-xs text-slate-300">
      <input type="checkbox" bind:checked={filterError} onchange={() => refresh(true)} />
      errors only
    </label>
    <label class="flex items-center gap-2 flex-1">
      <span class="text-xs text-slate-400">search</span>
      <input
        class="input text-xs py-1"
        placeholder="URL / model / error / body text (LIKE)"
        bind:value={searchQuery}
        onkeydown={(e) => e.key === 'Enter' && runSearch()}
      />
      <button class="btn btn-ghost text-xs" onclick={runSearch}>search</button>
      {#if searchQuery}
        <button class="btn btn-ghost text-xs" onclick={() => { searchQuery = ''; refresh(true); }}>
          clear
        </button>
      {/if}
    </label>
  </div>

  <!-- list + detail -->
  <div class="grid grid-cols-1 lg:grid-cols-[minmax(0,55fr)_minmax(0,45fr)] gap-3">
    <div class="card p-0 overflow-hidden flex flex-col" style="max-height: 75vh;">
      <div class="grid grid-cols-[7rem_8rem_1fr_4rem_5rem_6rem] gap-2 px-3 py-1.5
                  text-[10px] uppercase tracking-wider text-slate-500 border-b border-slate-800
                  bg-slate-900/80">
        <span>time</span>
        <span>backend</span>
        <span>path / preview</span>
        <span>status</span>
        <span>latency</span>
        <span>tokens</span>
      </div>
      <div class="overflow-y-auto flex-1">
        {#if items.length === 0 && !listBusy}
          <div class="px-3 py-6 text-center text-slate-500 text-sm">
            No upstream requests yet. Run a generate / inspect to populate this list.
          </div>
        {/if}
        {#each items as log (log.id)}
          <LogRow {log} selected={selectedId === log.id} onSelect={openDetail} />
        {/each}
        {#if hasMore}
          <button class="btn btn-ghost w-full text-xs mt-1" onclick={loadMore} disabled={listBusy}>
            {listBusy ? 'loading…' : `load more (older than ${nextCursor})`}
          </button>
        {/if}
      </div>
    </div>

    <div class="card p-0 overflow-hidden flex flex-col" style="max-height: 75vh;">
      {#if detailBusy}
        <div class="px-3 py-6 text-center text-slate-500 text-sm">loading…</div>
      {:else if selectedDetail}
        {@const d = selectedDetail}
        <div class="px-3 py-2 border-b border-slate-800 bg-slate-900/80 flex flex-wrap gap-2 items-center text-xs">
          <span class="font-mono text-slate-300">{d.method}</span>
          <span class="font-mono text-slate-400 truncate max-w-md" title={d.upstream_url}>
            {d.upstream_url}
          </span>
          <span class="chip">{d.backend_name}</span>
          {#if d.provider_name}<span class="chip">{d.provider_name}</span>{/if}
          {#if d.is_streaming}<span class="chip text-violet-300 border-violet-500/40">streaming</span>{/if}
          <span class="flex-1"></span>
          <span class="font-mono text-slate-400" title="status">
            {d.response_status_code ?? '—'}
          </span>
          <span class="font-mono text-slate-400" title="latency">
            {formatMs(d.latency_ms)}
          </span>
          {#if d.ttft_ms !== null}
            <span class="font-mono text-slate-400" title="time-to-first-token">
              ttft {formatMs(d.ttft_ms)}
            </span>
          {/if}
        </div>
        <div class="overflow-y-auto flex-1 p-3 space-y-3 text-xs">
          {#if d.model_resolved}
            <div>
              <span class="text-slate-500">model:</span>
              <span class="font-mono text-slate-300">{d.model_resolved}</span>
            </div>
          {/if}
          {#if d.error_message}
            <div class="text-rose-300 font-mono whitespace-pre-wrap">{d.error_message}</div>
          {/if}
          {#if d.completion_text}
            <details open class="border-l-2 border-emerald-500/30 pl-2">
              <summary class="cursor-pointer text-emerald-300 text-xs">completion</summary>
              <pre class="text-slate-200 whitespace-pre-wrap font-mono text-xs mt-1">{d.completion_text}</pre>
            </details>
          {/if}
          <details open class="border-l-2 border-sky-500/30 pl-2">
            <summary class="cursor-pointer text-sky-300 text-xs">request</summary>
            <div class="mt-2">
              <div class="text-slate-500 mb-1">headers</div>
              <JsonView data={d.request_headers ?? {}} name={null} />
              <div class="text-slate-500 mt-2 mb-1">body</div>
              {#if d.request_body !== null && d.request_body !== undefined}
                <JsonView data={d.request_body} name={null} />
              {:else if d.request_body_text}
                <pre class="text-slate-300 whitespace-pre-wrap">{d.request_body_text}</pre>
              {:else}
                <span class="text-slate-500 italic">(empty)</span>
              {/if}
            </div>
          </details>
          <details open class="border-l-2 border-amber-500/30 pl-2">
            <summary class="cursor-pointer text-amber-300 text-xs">response</summary>
            <div class="mt-2">
              <div class="text-slate-500 mb-1">headers</div>
              <JsonView data={d.response_headers ?? {}} name={null} />
              <div class="text-slate-500 mt-2 mb-1">body</div>
              {#if d.response_body !== null && d.response_body !== undefined}
                <JsonView data={d.response_body} name={null} />
              {:else if d.response_body_text}
                <pre class="text-slate-300 whitespace-pre-wrap">{d.response_body_text}</pre>
              {:else}
                <span class="text-slate-500 italic">(empty)</span>
              {/if}
            </div>
          </details>
          {#if d.stream_chunks && d.stream_chunks.length > 0}
            <details class="border-l-2 border-violet-500/30 pl-2">
              <summary class="cursor-pointer text-violet-300 text-xs">
                stream chunks ({d.stream_chunks.length})
              </summary>
              <div class="mt-2">
                <JsonView data={d.stream_chunks} name={null} open={false} />
              </div>
            </details>
          {/if}
        </div>
      {:else}
        <div class="px-3 py-6 text-center text-slate-500 text-sm">
          Select a row on the left to see its full request / response.
        </div>
      {/if}
    </div>
  </div>
</div>
