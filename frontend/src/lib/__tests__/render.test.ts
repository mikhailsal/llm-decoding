/**
 * Vitest parity tests for the visible-whitespace token renderer.
 *
 * The Python rendering rules live in :mod:`dsbx.cli.render`;
 * the same logic in TypeScript must produce the same visible markers so
 * the browser table reads identically to the TUI table.
 */

import { describe, it, expect } from 'vitest';
import {
  confidenceClass,
  fmtProb,
  isSpecialText,
  probWidth,
  renderTokenPlain,
  renderTokenSegments
} from '../render';

describe('renderTokenSegments', () => {
  it('marks empty token text explicitly', () => {
    expect(renderTokenSegments('')).toEqual([{ kind: 'empty', text: '<empty>' }]);
  });

  it('renders leading and trailing spaces as ␣ markers', () => {
    // Each space marker is ␣ + a zero-width space (U+200B) so long runs can
    // still wrap and be selected; assert against the real marker.
    const S = '\u2423\u200B';
    expect(renderTokenPlain(' I ')).toBe(`${S}I${S}`);
    expect(renderTokenPlain('  Paris')).toBe(`${S}${S}Paris`);
  });

  it('leaves internal spaces alone', () => {
    expect(renderTokenPlain('hello world')).toBe('hello world');
  });

  it('rewrites newline and tab inline', () => {
    expect(renderTokenPlain('a\nb')).toBe('a\u21B5b');
    expect(renderTokenPlain('a\tb')).toBe('a\u2192b');
  });

  it('marks an explicitly-special token as <special> even when its text is empty', () => {
    // Regression: backends that detokenize control tokens to "" must NOT
    // render the dim <empty> placeholder (which reads as "model emitted
    // nothing"); the is_special flag takes priority over the empty check.
    const segs = renderTokenSegments('', true);
    expect(segs[0].kind).toBe('special');
    expect(segs[0].text).toBe('<special>');
  });

  it('still renders a genuinely empty, non-special token as <empty>', () => {
    expect(renderTokenSegments('', false)).toEqual([{ kind: 'empty', text: '<empty>' }]);
  });

  it('renders <|endoftext|> as a special token via the heuristic', () => {
    expect(isSpecialText('<|endoftext|>')).toBe(true);
    expect(renderTokenSegments('<|endoftext|>')[0].kind).toBe('special');
  });

  it('recognizes DeepSeek fullwidth-pipe specials (<｜begin▁of▁sentence｜>)', () => {
    const ds = '<\uFF5Cbegin\u2581of\u2581sentence\uFF5C>';
    expect(isSpecialText(ds)).toBe(true);
    expect(renderTokenSegments(ds)[0].kind).toBe('special');
  });

  it('renders other control bytes as \\xNN', () => {
    expect(renderTokenPlain('a\x07b')).toBe('a\\x07b');
  });
});

describe('fmtProb / confidenceClass / probWidth', () => {
  it('formats fractional probabilities as percentages with 2 decimals', () => {
    expect(fmtProb(0.1234)).toBe('12.34%');
    expect(fmtProb(1.0)).toBe('100.00%');
  });

  it('renders nulls/NaN as ?', () => {
    expect(fmtProb(null)).toBe('?');
    expect(fmtProb(NaN)).toBe('?');
  });

  it('buckets confidence into five tailwind classes', () => {
    expect(confidenceClass(0.95)).toBe('bg-emerald-500');
    expect(confidenceClass(0.6)).toBe('bg-sky-500');
    expect(confidenceClass(0.3)).toBe('bg-amber-500');
    expect(confidenceClass(0.15)).toBe('bg-orange-500');
    expect(confidenceClass(0.05)).toBe('bg-rose-500');
    expect(confidenceClass(null)).toBe('bg-slate-700');
  });

  it('clamps probWidth into [0, 100]', () => {
    expect(probWidth(0.5)).toBe(50);
    expect(probWidth(-0.5)).toBe(0);
    expect(probWidth(1.5)).toBe(100);
    expect(probWidth(null)).toBe(0);
  });
});
