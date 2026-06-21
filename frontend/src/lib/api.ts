/**
 * Thin client for the dsbx web middleware. Every request attaches the
 * bearer token from the ``auth`` store; streaming endpoints are exposed
 * via ``apiStream`` which returns an async iterable of SSE frames.
 *
 * Error model: any non-2xx response is turned into an ``ApiError`` whose
 * ``message`` is the server's ``{detail: ...}`` payload when present. The
 * UI displays that message in a toast; the underlying status code is
 * exposed so call sites can branch on ``401`` (token rotation) /
 * ``403`` / ``404``.
 */

import { get } from 'svelte/store';
import { auth } from './stores/auth';
import { streamSseEvents, type SseFrame } from './sse';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

function authHeaders(): HeadersInit {
  const token = get(auth).token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function readError(res: Response): Promise<string> {
  try {
    const data = (await res.clone().json()) as { detail?: string };
    if (data && typeof data.detail === 'string') return data.detail;
  } catch {
    // not JSON
  }
  try {
    const text = await res.text();
    return text || `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

/** Plain GET/POST helper for non-streaming endpoints. */
export async function apiFetch<T>(
  path: string,
  init: RequestInit = {}
): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...authHeaders(),
      ...(init.headers || {})
    }
  });
  if (!res.ok) {
    const message = await readError(res);
    throw new ApiError(res.status, message);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/**
 * Open an authenticated SSE stream. Returns an async iterable of frames
 * AND a cancel callback so the caller can abort the connection when the
 * user navigates away mid-stream.
 *
 * Usage:
 *   const { events, cancel } = apiStream('/api/v1/generate/stream', body);
 *   for await (const evt of events) { ... }
 */
export function apiStream(
  path: string,
  body: unknown
): { events: AsyncIterable<SseFrame>; cancel: () => void } {
  const controller = new AbortController();
  const eventStream = (async function* () {
    const res = await fetch(path, {
      method: 'POST',
      signal: controller.signal,
      headers: {
        Accept: 'text/event-stream',
        'Content-Type': 'application/json',
        ...authHeaders()
      },
      body: JSON.stringify(body)
    });
    if (!res.ok) {
      const message = await readError(res);
      throw new ApiError(res.status, message);
    }
    if (!res.body) {
      throw new ApiError(500, 'no response body from streaming endpoint');
    }
    const reader = res.body.getReader();
    try {
      yield* streamSseEvents(reader);
    } finally {
      try {
        await reader.cancel();
      } catch {
        // already cancelled
      }
    }
  })();
  return {
    events: eventStream,
    cancel: () => controller.abort()
  };
}
