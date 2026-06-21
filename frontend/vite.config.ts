import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

/**
 * Vite + SvelteKit + Vitest. The proxy in dev mode lets ``pnpm dev``'s
 * Vite server forward every ``/api/v1/*`` request to ``dsbx web`` running
 * on :8765, so the same-origin Bearer token path works without CORS in
 * day-to-day development.
 */
export default defineConfig({
  plugins: [sveltekit()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
        // SSE responses need streaming to be pass-through; Vite handles
        // this automatically when the upstream Content-Type is
        // text/event-stream.
        ws: false
      }
    }
  },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node'
  }
});
