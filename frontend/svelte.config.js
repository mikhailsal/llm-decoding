import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    // adapter-static writes a fully prerendered bundle into ``build/``.
    // FastAPI mounts that directory at ``/`` so a single ``dsbx web`` process
    // serves both the API and the SPA. ``fallback`` makes client-side
    // routing work for direct loads of /inspect, /generate, etc.
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      fallback: 'index.html',
      precompress: false,
      strict: false
    })
  }
};

export default config;
