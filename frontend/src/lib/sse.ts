/**
 * Tiny SSE parser used to drain ``fetch``-based streaming responses.
 *
 * The dsbx middleware emits one event per ``data: <json>`` line followed
 * by a blank line. We don't use the browser's built-in ``EventSource``
 * because it can't attach a bearer header (a known omission of the API).
 * Instead we read the response body with ``ReadableStream.getReader`` and
 * feed each chunk into this parser.
 *
 * The shape mirrors the Python parser in
 * ``decoding_sandbox/backends/remote.py`` (``_iter_sse_events``):
 * comments (lines starting with ``:``) are dropped, unknown SSE fields
 * (``event:`` / ``id:`` / ``retry:``) are ignored, only the ``data:``
 * field is JSON-decoded.
 */

export interface SseFrame {
  event?: string;
  [k: string]: unknown;
}

/**
 * Pure split-on-double-newline parser, easiest to unit-test. Pass it the
 * full body of a non-streaming text response (e.g. in tests) and it
 * yields each ``data:`` payload as a parsed object.
 */
export function parseSseBody(body: string): SseFrame[] {
  const out: SseFrame[] = [];
  for (const chunk of body.split(/\r?\n\r?\n/)) {
    if (!chunk.trim()) continue;
    const frame = parseSseFrame(chunk);
    if (frame) out.push(frame);
  }
  return out;
}

/**
 * Async iterator that consumes a ``ReadableStream<Uint8Array>`` (the body
 * of an authenticated ``fetch`` to a streaming endpoint) and yields one
 * decoded SSE frame at a time. The caller is responsible for cancelling
 * the underlying response if the iteration is aborted early.
 *
 * The internal buffer is kept small: we emit frames as soon as the
 * separating blank line arrives, mirroring the cadence of the server.
 */
export async function* streamSseEvents(
  reader: ReadableStreamDefaultReader<Uint8Array>
): AsyncGenerator<SseFrame, void, void> {
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (value) {
      buffer += decoder.decode(value, { stream: true });
      let idx;
      // SSE frame boundary is a blank line. ``\n\n`` is canonical; some
      // upstream proxies normalize line endings to ``\r\n``, so accept both.
      while (true) {
        const nlnl = buffer.indexOf('\n\n');
        const crnlcrnl = buffer.indexOf('\r\n\r\n');
        if (nlnl === -1 && crnlcrnl === -1) break;
        if (nlnl !== -1 && (crnlcrnl === -1 || nlnl < crnlcrnl)) {
          idx = nlnl;
        } else {
          idx = crnlcrnl;
        }
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + (frame.includes('\r\n') ? 4 : 2));
        const parsed = parseSseFrame(frame);
        if (parsed) yield parsed;
      }
    }
    if (done) break;
  }
  // Flush trailing buffer if the server didn't terminate with a blank
  // line (it always does today, but defensive parsing is cheap).
  if (buffer.trim()) {
    const parsed = parseSseFrame(buffer);
    if (parsed) yield parsed;
  }
}

function parseSseFrame(raw: string): SseFrame | null {
  const dataLines: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (!line || line.startsWith(':')) continue;
    if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).replace(/^\s+/, ''));
    }
  }
  if (dataLines.length === 0) return null;
  try {
    return JSON.parse(dataLines.join('\n')) as SseFrame;
  } catch {
    return null;
  }
}
