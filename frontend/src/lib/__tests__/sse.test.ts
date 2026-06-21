/**
 * Vitest unit tests for the SSE parser.
 *
 * We use ``parseSseBody`` for the easy "give me a string" path and a
 * tiny mock ``ReadableStream`` for ``streamSseEvents`` to exercise the
 * incremental buffering that real streaming responses depend on.
 */

import { describe, it, expect } from 'vitest';
import { parseSseBody, streamSseEvents, type SseFrame } from '../sse';

function mockReader(chunks: string[]): ReadableStreamDefaultReader<Uint8Array> {
  const enc = new TextEncoder();
  const queue = chunks.map((c) => enc.encode(c));
  let i = 0;
  return {
    async read() {
      if (i >= queue.length) return { value: undefined, done: true };
      const value = queue[i++];
      return { value, done: false };
    },
    cancel: async () => {},
    closed: Promise.resolve(undefined),
    releaseLock() {}
  } as unknown as ReadableStreamDefaultReader<Uint8Array>;
}

describe('parseSseBody', () => {
  it('returns one frame per data: chunk', () => {
    const body =
      'data: {"event":"step","n":1}\n\ndata: {"event":"step","n":2}\n\ndata: {"event":"done"}\n\n';
    const frames = parseSseBody(body);
    expect(frames.length).toBe(3);
    expect(frames[0]).toMatchObject({ event: 'step', n: 1 });
    expect(frames[2]).toMatchObject({ event: 'done' });
  });

  it('ignores comments and unknown SSE fields', () => {
    const body =
      ': heartbeat\n\nevent: ignored\ndata: {"a":1}\n\nid: 42\ndata: {"a":2}\n\n';
    const frames = parseSseBody(body);
    expect(frames.map((f) => (f as { a: number }).a)).toEqual([1, 2]);
  });

  it('drops malformed frames without aborting', () => {
    const body =
      'data: {"good":1}\n\ndata: not-json\n\ndata: {"good":2}\n\n';
    const frames = parseSseBody(body);
    expect(frames.length).toBe(2);
  });
});

describe('streamSseEvents', () => {
  async function drain(reader: ReadableStreamDefaultReader<Uint8Array>): Promise<SseFrame[]> {
    const out: SseFrame[] = [];
    for await (const frame of streamSseEvents(reader)) {
      out.push(frame);
    }
    return out;
  }

  it('emits frames as their blank-line terminator arrives', async () => {
    // Three chunks that don't align with frame boundaries.
    const reader = mockReader([
      'data: {"event":"step","n":1}\n',
      '\ndata: {"event":',
      '"step","n":2}\n\ndata: {"event":"done"}\n\n'
    ]);
    const frames = await drain(reader);
    expect(frames).toEqual([
      { event: 'step', n: 1 },
      { event: 'step', n: 2 },
      { event: 'done' }
    ]);
  });

  it('handles CRLF line endings as well as LF', async () => {
    const reader = mockReader([
      'data: {"event":"step","n":1}\r\n\r\ndata: {"event":"done"}\r\n\r\n'
    ]);
    const frames = await drain(reader);
    expect(frames).toEqual([
      { event: 'step', n: 1 },
      { event: 'done' }
    ]);
  });

  it('flushes a trailing buffer without final blank line', async () => {
    const reader = mockReader(['data: {"event":"step","n":1}']);
    const frames = await drain(reader);
    expect(frames).toEqual([{ event: 'step', n: 1 }]);
  });
});
