// Prerender every route -- adapter-static needs this to produce a flat
// build directory we can mount inside FastAPI. SSR is disabled because
// the app is purely browser-side (it talks to the middleware via fetch).
export const prerender = true;
export const ssr = false;
