"""Example custom decoding function for `dsbx generate --sampler custom`.

A sampler receives the ranked candidate list and a context, and returns either a
token id (int) or a fully-built SamplerDecision. This toy "contrarian" sampler
picks the SECOND most likely token (when available) so you can clearly see it
diverge from greedy -- a deliberately simple template to copy.

Run:
  dsbx generate "Once upon a time" --backend hf \
      --sampler custom --custom-file examples/custom_sampler.py:decode
"""

from __future__ import annotations


def decode(cands, ctx) -> int:
    """Pick the runner-up token, except on the very first step (use greedy)."""
    if ctx.step == 0 or len(cands) < 2:
        return cands[0].token_id
    return cands[1].token_id
