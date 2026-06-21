/**
 * Bearer-token store. The token is persisted in ``localStorage`` so a
 * reload doesn't kick the user out of an in-progress decoding session.
 *
 * The store is intentionally tiny -- the actual auth check happens on the
 * server with every request. Logout clears the value; the login page
 * then redirects to ``/`` once the user pastes a new token.
 */

import { writable } from 'svelte/store';
import { browser } from '$app/environment';

const KEY = 'dsbx_web_token';

export interface AuthState {
  token: string | null;
}

function read(): AuthState {
  if (!browser) return { token: null };
  try {
    return { token: window.localStorage.getItem(KEY) };
  } catch {
    return { token: null };
  }
}

function persist(state: AuthState) {
  if (!browser) return;
  try {
    if (state.token) {
      window.localStorage.setItem(KEY, state.token);
    } else {
      window.localStorage.removeItem(KEY);
    }
  } catch {
    // localStorage can be disabled (private mode); we keep the state in
    // memory only -- the UI still works for the current tab.
  }
}

const initial = read();
const store = writable<AuthState>(initial);

export const auth = {
  subscribe: store.subscribe,
  setToken(token: string) {
    const state = { token: token.trim() || null };
    persist(state);
    store.set(state);
  },
  logout() {
    persist({ token: null });
    store.set({ token: null });
  }
};
