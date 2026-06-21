<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { auth } from '$lib/stores/auth';
  import { info } from '$lib/stores/info';

  let { children } = $props();

  $effect(() => {
    if (!$auth.token && $page.url.pathname !== '/') {
      goto('/');
    }
  });

  onMount(() => {
    if ($auth.token && !$info.info) {
      info.refresh();
    }
  });

  const navLinks: { href: string; label: string }[] = [
    { href: '/inspect', label: 'Inspect' },
    { href: '/generate', label: 'Generate' },
    { href: '/manual', label: 'Manual' },
    { href: '/spec', label: 'Speculative' },
    { href: '/status', label: 'Status' }
  ];

  function logout() {
    auth.logout();
    goto('/');
  }
</script>

<div class="min-h-full flex flex-col">
  <header class="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-10">
    <div class="max-w-7xl mx-auto px-4 py-2 flex items-center gap-3">
      <a href="/" class="text-sky-400 font-bold text-lg tracking-tight">dsbx</a>
      <span class="text-slate-500 text-sm font-mono">
        {$info.info ? $info.info.server_label : 'web'}
      </span>
      <nav class="flex-1 flex items-center gap-1">
        {#if $auth.token}
          {#each navLinks as link}
            {@const active = $page.url.pathname.startsWith(link.href)}
            <a
              href={link.href}
              class="px-3 py-1.5 rounded-md text-sm transition-colors {active
                ? 'bg-sky-500/20 text-sky-300'
                : 'text-slate-300 hover:bg-slate-800'}"
            >
              {link.label}
            </a>
          {/each}
        {/if}
      </nav>
      {#if $auth.token}
        <span class="text-xs text-slate-500 font-mono">
          token: {$auth.token.slice(0, 4)}…{$auth.token.slice(-3)}
        </span>
        <button class="btn btn-ghost text-xs" onclick={logout}>logout</button>
      {/if}
    </div>
  </header>

  <main class="flex-1">
    <div class="max-w-7xl mx-auto px-4 py-6">
      {@render children?.()}
    </div>
  </main>

  <footer class="border-t border-slate-800 text-center py-3 text-xs text-slate-500">
    {#if $info.info}
      engine {$info.info.engine_version} · default backend
      <span class="font-mono text-slate-400">{$info.info.default_backend}</span>
    {:else}
      dsbx web · all keys live on the middleware
    {/if}
  </footer>
</div>
