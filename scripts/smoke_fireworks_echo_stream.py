"""Phase 5 smoke test: does Fireworks /v1/completions tolerate echo+stream?

We want to know whether a single call with ``echo=True``, ``stream=True``,
``max_tokens>0``, and ``logprobs=N`` actually returns BOTH the per-prompt
position logprobs AND the per-generated-token logprobs, in a useful order.
If it does, we can collapse the current two-request "include prompt"
workflow (``score_prompt`` + ``stream_native``) into a single network round
trip and roughly halve the HTTP cost of the inspect / generate-with-prompt
modes.

The script prints the *shape* of each SSE chunk: which choice index, whether
``logprobs`` is set, the count of token entries / top-logprob entries, and
the first few token strings. That's enough to decide:

* If chunk 0 carries ALL prompt-position logprobs at once, the combined path
  is trivially safe -- we just split the first chunk into a synthetic
  ``score_prompt`` shape and treat the rest as the normal stream.
* If prompt logprobs come spread across chunks 0..N (one per fragment),
  we'd need a small assembler in :class:`OpenAICompatBackend`.
* If the server returns prompt logprobs only at the END (or not at all
  for streaming), we keep the two-request path.

Run:

    FIREWORKS_API_KEY=... python scripts/smoke_fireworks_echo_stream.py

By default it hits ``accounts/fireworks/models/llama-v3p1-8b-instruct``;
override with ``--model``. ``--prompt`` swaps the test prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx


def _iter_sse(resp: httpx.Response):
    """Yield decoded JSON payloads from an OpenAI-style SSE stream."""
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
        data = line[5:].strip() if line.startswith("data:") else line.strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="accounts/fireworks/models/llama-v3p1-8b-instruct",
    )
    ap.add_argument(
        "--prompt",
        default="The capital of France is",
    )
    ap.add_argument("--max-tokens", type=int, default=5)
    ap.add_argument("--top-logprobs", type=int, default=5)
    ap.add_argument(
        "--base-url",
        default="https://api.fireworks.ai/inference/v1",
        help="override for staging / proxies",
    )
    args = ap.parse_args()

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("FIREWORKS_API_KEY not set", file=sys.stderr)
        return 2

    body: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": int(args.max_tokens),
        "stream": True,
        "echo": True,
        # NewLogProbs shape: logprobs=true + top_logprobs=N. This is the
        # format the sandbox migrated to in Phase 3; we want to confirm
        # it round-trips when combined with echo+stream.
        "logprobs": True,
        "top_logprobs": int(args.top_logprobs),
        # Diagnostics so we can compare chunk counts vs server-side
        # generation-duration to spot prompt-fragment timing.
        "perf_metrics_in_response": True,
        "raw_output": True,
        "sampling_mask": "count",
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }

    print(f"--> POST {args.base_url}/completions")
    print(f"    model: {args.model}")
    print(f"    prompt: {args.prompt!r}  max_tokens: {args.max_tokens}")
    print(f"    body keys: {sorted(body.keys())}")
    print()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    chunk_no = 0
    prompt_token_chunks: list[int] = []  # chunk indices that carry prompt logprobs
    completion_token_chunks: list[int] = []
    with (
        httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0)) as client,
        client.stream(
            "POST",
            f"{args.base_url}/completions",
            json=body,
            headers=headers,
        ) as resp,
    ):
        print(f"<-- {resp.status_code} {resp.reason_phrase}")
        if resp.status_code >= 400:
            # Buffer and dump the body so we know if the server
            # rejected echo+stream with a 400.
            body_bytes = b"".join(resp.iter_raw())
            print(body_bytes.decode("utf-8", "replace"))
            return 1
        for chunk in _iter_sse(resp):
            chunk_no += 1
            choices = chunk.get("choices") or []
            if not choices:
                {k: v for k, v in chunk.items() if k in {"usage", "perf_metrics", "raw_output"}}
                print(f"chunk #{chunk_no:02d} [no choices] keys={list(chunk.keys())}")
                for k in ("usage", "perf_metrics", "raw_output"):
                    if k in chunk:
                        print(f"  {k}: {json.dumps(chunk[k])[:200]}")
                continue
            ch = choices[0]
            lp = ch.get("logprobs") or {}
            content = lp.get("content") or []
            # ``content`` carries one entry per emitted-or-echoed
            # position. Each entry has ``token``, optional
            # ``token_id``, ``logprob``, ``top_logprobs``, and
            # ``sampling_mask_count`` (when requested).
            tokens = [str(e.get("token", "")) for e in content]
            ids = [e.get("token_id") for e in content]
            # Heuristic: prompt-echo entries (the first N positions
            # of the first chunk) typically have a non-empty
            # top_logprobs that includes the actual prompt token at
            # rank 0; emitted positions do too but they appear
            # incrementally chunk-by-chunk. Print enough to
            # reconstruct ordering visually.
            print(
                f"chunk #{chunk_no:02d}  positions={len(content)}  "
                f"finish={ch.get('finish_reason')!r}  text={ch.get('text')!r}"
            )
            for i, (tok, tid) in enumerate(zip(tokens, ids, strict=False)):
                top = content[i].get("top_logprobs") or []
                print(f"   pos {i:>3}: id={tid!r:>8} token={tok!r:<14} top_n={len(top)}")
            if ch.get("text"):
                # Heuristic to bucket chunks: if the chunk's ``text``
                # equals the full prompt + echoed continuation OR
                # carries many positions at once, treat as
                # prompt-echo; otherwise as completion stream.
                if chunk_no == 1 and len(content) > 1:
                    prompt_token_chunks.append(chunk_no)
                else:
                    completion_token_chunks.append(chunk_no)
    print()
    print("=== summary ===")
    print(f"total chunks: {chunk_no}")
    print(f"prompt-echo chunks (heuristic):     {prompt_token_chunks}")
    print(f"completion-stream chunks (heuristic): {completion_token_chunks}")
    print()
    print("Decision criteria for combined path implementation:")
    print(" - if chunk #1 carries len(prompt) positions and rest carry 1 each:")
    print("   -> SAFE to implement stream_native_with_echo (split chunk #1)")
    print(" - if positions arrive interleaved or only at the end:")
    print("   -> KEEP the two-request path (current behavior)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
