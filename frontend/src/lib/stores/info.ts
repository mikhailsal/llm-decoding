/**
 * Cached ``GET /api/v1/info`` payload.
 *
 * The header + the backend dropdowns on every page need this; we fetch it
 * once after login and refresh on demand. A failure clears the token (the
 * user is told to log in again) which is the cleanest UX for "401 on the
 * very first authenticated request".
 */

import { writable, get as svelteGet } from 'svelte/store';
import { ApiError, apiFetch } from '../api';
import { auth } from './auth';
import type { InfoResponse } from '../types';

interface InfoState {
  info: InfoResponse | null;
  loading: boolean;
  error: string | null;
  /**
   * The backend the user has currently selected from the dropdown. We
   * default it to the server's default_backend after the first fetch.
   */
  selected: string | null;
}

const store = writable<InfoState>({
  info: null,
  loading: false,
  error: null,
  selected: null
});

export const info = {
  subscribe: store.subscribe,

  async refresh(): Promise<void> {
    store.update((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await apiFetch<InfoResponse>('/api/v1/info');
      store.update((s) => ({
        info: data,
        loading: false,
        error: null,
        // Keep an existing selection unless it's no longer in the listing.
        selected:
          s.selected && data.backends.some((b) => b.name === s.selected)
            ? s.selected
            : data.default_backend
      }));
    } catch (exc) {
      if (exc instanceof ApiError && exc.status === 401) {
        auth.logout();
      }
      store.update((s) => ({
        ...s,
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc)
      }));
    }
  },

  select(name: string): void {
    store.update((s) => ({ ...s, selected: name }));
  },

  selectedBackend(): string | null {
    return svelteGet(store).selected;
  }
};
