/**
 * Visible-whitespace token rendering.
 *
 * Mirrors :mod:`decoding_sandbox.cli.render` so the browser shows the same
 * markers the TUI does:
 *
 *   - leading / trailing space ->  ``␣``  (one per character)
 *   - newline                  ->  ``↵``
 *   - tab                      ->  ``→``
 *   - empty string             ->  ``<empty>`` (dim italic)
 *   - special token (EOS/BOS/PAD/<|...|>) -> magenta + ``<special>`` marker
 *     when the printable form is empty.
 *
 * The output is a list of ``{text, kind}`` segments so the renderer can
 * wrap each part in a ``<span>`` with the right Tailwind class -- this is
 * what makes a row containing ``"I"``, ``" I"`` and ``"I "`` actually
 * distinguishable at a glance.
 */

export type TokenSegmentKind = 'plain' | 'ws' | 'special' | 'empty' | 'control';

export interface TokenSegment {
  kind: TokenSegmentKind;
  text: string;
}

const SPACE = '\u2423\u200B'; // ␣ + zero-width space
const NEWLINE = '\u21B5'; // ↵
const TAB = '\u2192'; // →

export function isSpecialText(text: string): boolean {
  if (!text) return false;
  return /^<\|[^|]*\|>$/.test(text);
}

/**
 * Decompose a token's text into segments. ``isSpecial`` overrides the
 * heuristic (callers should pass the backend's ``is_special`` flag when
 * they have it).
 */
export function renderTokenSegments(
  text: string,
  isSpecial = false
): TokenSegment[] {
  if (text === '') {
    return [{ kind: 'empty', text: '<empty>' }];
  }
  if (isSpecial || isSpecialText(text)) {
    if (text.trim() === '') {
      return [{ kind: 'special', text: '<special>' }];
    }
    return [{ kind: 'special', text }];
  }
  const out: TokenSegment[] = [];
  // Leading spaces
  let i = 0;
  while (i < text.length && text[i] === ' ') {
    out.push({ kind: 'ws', text: SPACE });
    i++;
  }
  // Middle: rewrite \n / \t inline but leave internal spaces alone.
  let middleEnd = text.length;
  while (middleEnd > i && text[middleEnd - 1] === ' ') {
    middleEnd--;
  }
  const middle = text.slice(i, middleEnd);
  if (middle.length > 0) {
    let buf = '';
    for (const ch of middle) {
      if (ch === '\n') {
        if (buf) {
          out.push({ kind: 'plain', text: buf });
          buf = '';
        }
        out.push({ kind: 'ws', text: NEWLINE });
      } else if (ch === '\t') {
        if (buf) {
          out.push({ kind: 'plain', text: buf });
          buf = '';
        }
        out.push({ kind: 'ws', text: TAB });
      } else if (ch.charCodeAt(0) < 0x20 && ch !== ' ') {
        if (buf) {
          out.push({ kind: 'plain', text: buf });
          buf = '';
        }
        out.push({
          kind: 'control',
          text: '\\x' + ch.charCodeAt(0).toString(16).padStart(2, '0')
        });
      } else {
        buf += ch;
      }
    }
    if (buf) out.push({ kind: 'plain', text: buf });
  }
  // Trailing spaces
  for (let j = middleEnd; j < text.length; j++) {
    out.push({ kind: 'ws', text: SPACE });
  }
  return out;
}

/** Plain-text variant (no markup) for ``alt`` text and ``aria-label``. */
export function renderTokenPlain(text: string, isSpecial = false): string {
  return renderTokenSegments(text, isSpecial)
    .map((s) => s.text)
    .join('');
}

/** Format a probability for tables -- mirrors ``cli/render.fmt_prob``. */
export function fmtProb(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return '?';
  // Same honesty contract as ``formatProbPct``: a token with a
  // genuine but tiny probability (e.g. the chosen one falling at
  // rank 247 with p≈1e-7) should NOT render as ``0.00%`` -- that
  // reads like the model assigned zero mass, which is a lie. Show
  // ``<0.1%`` instead so the student sees "small but nonzero".
  if (p < 0.001) return '<0.1%';
  return (p * 100).toFixed(2) + '%';
}

/** Tailwind color class for a confidence bar -- 5 buckets, like the TUI. */
export function confidenceClass(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return 'bg-slate-700';
  if (p >= 0.8) return 'bg-emerald-500';
  if (p >= 0.5) return 'bg-sky-500';
  if (p >= 0.25) return 'bg-amber-500';
  if (p >= 0.1) return 'bg-orange-500';
  return 'bg-rose-500';
}

/**
 * Background-style tailwind class used to color *inline text* by its
 * probability bucket (e.g. one chosen token inside a paragraph of running
 * completion). Same five buckets as ``confidenceClass`` but at low opacity
 * so the underlying text stays readable against a dark surface. ``null`` /
 * NaN -> a neutral slate so unknown-prob tokens remain visually grouped
 * but don't draw the eye.
 */
export function tokenBackgroundClass(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) {
    return 'bg-slate-700/40 text-slate-200';
  }
  if (p >= 0.8) return 'bg-emerald-500/30 text-emerald-50';
  if (p >= 0.5) return 'bg-sky-500/30 text-sky-50';
  if (p >= 0.25) return 'bg-amber-500/30 text-amber-50';
  if (p >= 0.1) return 'bg-orange-500/30 text-orange-50';
  return 'bg-rose-500/30 text-rose-50';
}

/** ``exp(logprob)`` -> linear prob, clamped to [0,1]; null/NaN passes through. */
export function probFromLogprob(lp: number | null | undefined): number | null {
  if (lp === null || lp === undefined || !Number.isFinite(lp)) return null;
  return Math.max(0, Math.min(1, Math.exp(lp)));
}

/** Convenience helper for probability bars (width in 0..100). */
export function probWidth(p: number | null | undefined): number {
  if (p === null || p === undefined || !Number.isFinite(p)) return 0;
  return Math.max(0, Math.min(100, Math.round(p * 100)));
}

/**
 * Format a logprob as a human-readable percent, distinguishing three
 * very different situations the UI used to conflate as "0.0%":
 *
 *   - logprob is null / NaN  -> "?"      (the upstream gave us no data;
 *                                          e.g. Fireworks position 0 of
 *                                          an echo response carries no
 *                                          top_logprobs because the
 *                                          autoregressive model has
 *                                          nothing to predict from)
 *   - prob > 0 but < 0.1%    -> "<0.1%"  (the token IS in the
 *                                          distribution -- the sampler
 *                                          can and sometimes does pick
 *                                          it -- but it rounds down to
 *                                          0.0% at one decimal place;
 *                                          rendering "0.0%" reads as
 *                                          "impossible" which is
 *                                          actively misleading in a
 *                                          pedagogical sandbox)
 *   - prob >= 0.1%           -> "X.X%"   (normal display, 1 decimal)
 *
 * Returns the formatted string only; callers decide whether to render
 * tooltips with the raw logprob alongside.
 */
export function formatProbPct(lp: number | null | undefined): string {
  if (lp === null || lp === undefined || !Number.isFinite(lp)) return '?';
  const p = Math.exp(lp);
  if (p < 0.001) return '<0.1%';
  return (p * 100).toFixed(1) + '%';
}
