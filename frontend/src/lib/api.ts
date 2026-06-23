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
import type {
  LogDetail,
  LogListParams,
  LogListResponse,
  LogStats,
  RemoteStatus
} from './types';

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

// --------------------------------------------------------------------------- //
// Remote model control (swappable dsbx-serve model slot)
// --------------------------------------------------------------------------- //
/** Live model-slot state of a remote dsbx-serve host. */
export function getRemoteStatus(name: string): Promise<RemoteStatus> {
  return apiFetch<RemoteStatus>(
    `/api/v1/backends/${encodeURIComponent(name)}/status`
  );
}

/** Ask a remote host to (re)load ``model`` (null = its default). */
export function reloadRemoteModel(
  name: string,
  model: string | null
): Promise<RemoteStatus> {
  return apiFetch<RemoteStatus>(
    `/api/v1/backends/${encodeURIComponent(name)}/reload`,
    { method: 'POST', body: JSON.stringify({ model }) }
  );
}

// --------------------------------------------------------------------------- //
// Upstream-request log API
// --------------------------------------------------------------------------- //
function logsQueryString(params: LogListParams | undefined): string {
  if (!params) return '';
  const usp = new URLSearchParams();
  if (params.cursor) usp.set('cursor', params.cursor);
  if (params.limit !== undefined && params.limit !== null) {
    usp.set('limit', String(params.limit));
  }
  if (params.backend) usp.set('backend', params.backend);
  if (params.provider) usp.set('provider', params.provider);
  if (params.status_code !== undefined && params.status_code !== null) {
    usp.set('status_code', String(params.status_code));
  }
  if (params.is_error !== undefined && params.is_error !== null) {
    usp.set('is_error', params.is_error ? 'true' : 'false');
  }
  if (params.since) usp.set('since', params.since);
  const qs = usp.toString();
  return qs ? `?${qs}` : '';
}

/** Fetch one page of log rows, newest first. */
export function listLogs(params?: LogListParams): Promise<LogListResponse> {
  return apiFetch<LogListResponse>(`/api/v1/logs${logsQueryString(params)}`);
}

/** Fetch one full log row including bodies and stream chunks. */
export function getLog(id: string): Promise<LogDetail> {
  return apiFetch<LogDetail>(`/api/v1/logs/${encodeURIComponent(id)}`);
}

/** LIKE search across URL / model / error / body text columns. */
export function searchLogs(q: string, limit = 50): Promise<LogListResponse> {
  const usp = new URLSearchParams({ q, limit: String(limit) });
  return apiFetch<LogListResponse>(`/api/v1/logs/search?${usp.toString()}`);
}

/** Dashboard-style counters (totals, error count, average latency). */
export function getLogStats(): Promise<LogStats> {
  return apiFetch<LogStats>('/api/v1/logs/stats');
}

/** Delete one row, all rows older than ``before``, or every row. */
export function deleteLogs(opts: {
  log_id?: string;
  before?: string;
  all?: boolean;
}): Promise<{ deleted: number }> {
  const usp = new URLSearchParams();
  if (opts.log_id) usp.set('log_id', opts.log_id);
  if (opts.before) usp.set('before', opts.before);
  if (opts.all) usp.set('all', 'true');
  const qs = usp.toString();
  return apiFetch<{ deleted: number }>(
    `/api/v1/logs${qs ? `?${qs}` : ''}`,
    { method: 'DELETE' }
  );
}
