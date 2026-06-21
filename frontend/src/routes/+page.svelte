<script lang="ts">
  import { goto } from '$app/navigation';
  import { auth } from '$lib/stores/auth';
  import { info } from '$lib/stores/info';
  import { apiFetch, ApiError } from '$lib/api';

  let token = $state('');
  let busy = $state(false);
  let error = $state<string | null>(null);

  async function login(e: Event) {
    e.preventDefault();
    error = null;
    busy = true;
    auth.setToken(token);
    try {
      // Probe the API to validate the token before we redirect.
      await apiFetch('/api/v1/info');
      await info.refresh();
      goto('/status');
    } catch (exc) {
      auth.logout();
      if (exc instanceof ApiError && exc.status === 401) {
        error = 'token rejected by the middleware';
      } else {
        error = exc instanceof Error ? exc.message : String(exc);
      }
    } finally {
      busy = false;
    }
  }
</script>

<div class="max-w-md mx-auto mt-16 card">
  <h1 class="text-xl font-semibold text-slate-100 mb-1">dsbx web</h1>
  <p class="text-sm text-slate-400 mb-4">
    Paste the bearer token configured in <span class="font-mono">[web].api_token</span>
    or <span class="font-mono">$DSBX_WEB_TOKEN</span>.
  </p>
  <form onsubmit={login} class="space-y-3">
    <div>
      <label class="label" for="token">Token</label>
      <input
        id="token"
        type="password"
        class="input font-mono"
        bind:value={token}
        placeholder="paste the bearer token"
        autocomplete="current-password"
        required
      />
    </div>
    {#if error}
      <div class="text-sm text-rose-400">{error}</div>
    {/if}
    <button type="submit" class="btn btn-primary w-full" disabled={busy}>
      {busy ? 'verifying…' : 'log in'}
    </button>
  </form>
  <p class="text-xs text-slate-500 mt-4">
    Your token is stored in <span class="font-mono">localStorage</span> so a reload doesn't
    kick you out. The middleware authenticates every API call.
  </p>
</div>
