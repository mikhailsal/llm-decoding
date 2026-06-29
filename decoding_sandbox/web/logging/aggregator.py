"""SSE-stream aggregation for the logging transport.

The logging transport tees every byte the upstream sends; this module
turns those bytes into (a) a flat list of parsed SSE frames -- one per
``data: ...\\n\\n`` chunk -- and (b) a single "assembled" response body
that's shape-compatible with what a non-streaming call would have
returned.

We support two wire shapes:

1. **OpenAI-compat ``/chat/completions`` and ``/completions`` SSE.**
   Each frame is ``{"choices": [{"delta": {"content": "Hel"}, ...}], ...}``
   and the merge logic mirrors a production proxy gateway
   -- concatenate ``delta.content``, ``delta.reasoning_content``,
   deep-merge ``tool_calls`` list items, capture the trailing ``usage``
   block (some providers emit it on the last data frame, others as a
   sibling). ``[DONE]`` is recognized as a terminator and dropped from
   the chunks list.
2. **dsbx-native ``/v1/generate/stream`` and ``/v1/spec/stream`` SSE.**
   The wire shape (see ``decoding_sandbox/web/streaming.py``) is
   ``{"event": "step", "step": {...}}`` / ``{"event": "usage", ...}`` /
   ``{"event": "done", ...}``. We extract token counts from the ``usage``
   frame, join ``step.decision.token_text`` into a completion string,
   capture ``stop_reason`` / ``error`` from the terminating ``done``.

Dispatch happens in :func:`aggregate_stream`: the heuristic looks at the
first parsed frame for an ``event`` key (dsbx) vs ``choices`` key
(OpenAI). Anything we can't classify falls through to a neutral
aggregator that just hands back the raw chunks without a synthesized
response body, so the log still has SOMETHING usable to show.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# The set of keys whose string values get concatenated when merging
# consecutive deltas. Matches a production proxy gateway; refusal/reasoning_content show
# up on OpenAI-style providers, content is the bread and butter.
_STRING_MERGE_KEYS = frozenset({"content", "reasoning_content", "reasoning", "refusal"})

# Inside the list-of-dicts merge (tool_calls primarily), these inner keys
# are themselves string-concatenated rather than overwritten. ``arguments``
# is a JSON string the provider builds up token-by-token.
_LIST_ITEM_CONCAT_KEYS = frozenset({"arguments", "summary", "text"})


# --------------------------------------------------------------------------- #
# Shape: result of aggregating one stream
# --------------------------------------------------------------------------- #
@dataclass
class AggregatedStream:
    """Output of :func:`aggregate_stream` for one captured byte stream.

    Field semantics:

    - ``chunks``: every parsed SSE frame in order. Non-JSON frames and
      the ``[DONE]`` terminator are excluded; everything else lands here
      so the detail UI can replay the stream chunk-by-chunk.
    - ``assembled_body``: the single JSON document the caller would have
      seen if the same call had been non-streaming. For OpenAI shapes
      this synthesizes a ``{"choices": [{"message": {...}}], "usage": ...}``
      object; for dsbx shapes this is a ``{"events": [...], "completion": ...,
      "usage": ...}`` object. ``None`` when we couldn't make sense of the
      frames at all.
    - ``completion_text`` / ``stop_reason``: convenience denormalizations
      the logs UI uses to render the row at a glance without re-parsing
      ``assembled_body``.
    """

    chunks: list[Any] = field(default_factory=list)
    assembled_body: Any = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    completion_text: str | None = None
    stop_reason: str | None = None
    error_message: str | None = None
    model_resolved: str | None = None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def aggregate_stream(raw: bytes) -> AggregatedStream:
    """Parse ``raw`` SSE bytes and return an :class:`AggregatedStream`.

    Empty / non-SSE input returns an empty result (no chunks, no
    assembled body, no error). That's the right behaviour for the cases
    where the transport tee captures bytes that aren't actually a
    stream (e.g. a connect-time 502 that returned application/json on
    the same code path).
    """
    parsed_frames = _parse_sse_frames(raw)
    if not parsed_frames:
        return AggregatedStream()
    shape = _detect_shape(parsed_frames)
    if shape == "openai":
        return _merge_openai_stream(parsed_frames)
    if shape == "dsbx":
        return _merge_dsbx_stream(parsed_frames)
    return AggregatedStream(chunks=parsed_frames)


# --------------------------------------------------------------------------- #
# SSE parsing: bytes -> list of decoded JSON frames
# --------------------------------------------------------------------------- #
def _parse_sse_frames(raw: bytes) -> list[Any]:
    """Pull every ``data: <json>`` payload out of an SSE byte stream.

    Same shape as the parser in ``decoding_sandbox/backends/remote.py``:
    comment lines (leading ``:``) are dropped, ``[DONE]`` terminates the
    stream cleanly, anything that doesn't decode as JSON is silently
    skipped so a single malformed frame can't abandon the rest.
    """
    if not raw:
        return []
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return []

    frames: list[Any] = []
    # SSE separator is a blank line. httpx normalizes ``\r\n`` to ``\n``
    # for us but we accept both just in case the test feeds us raw
    # network bytes.
    chunks = text.split("\n\n") if "\n\n" in text else text.split("\r\n\r\n")
    for chunk in chunks:
        if not chunk.strip():
            continue
        data_lines: list[str] = []
        for line in chunk.splitlines():
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
        if not data_lines:
            continue
        payload = "\n".join(data_lines).strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            frames.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return frames


def _detect_shape(frames: list[Any]) -> str:
    """Decide between ``"openai"`` and ``"dsbx"`` based on first frame keys.

    A dsbx-native frame always carries an ``event`` discriminator we
    can match on directly. An OpenAI-style frame has ``choices`` (or
    ``object`` == ``"chat.completion.chunk"`` / ``"text_completion"``).
    Ambiguous frames default to ``"openai"`` because that's the more
    common upstream by a wide margin -- the user's research workflow
    is dominated by Fireworks / NIM / OpenRouter calls.
    """
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        if "event" in frame and isinstance(frame.get("event"), str):
            return "dsbx"
        if "choices" in frame or "object" in frame:
            return "openai"
    return "openai"


# --------------------------------------------------------------------------- #
# OpenAI-compat merge (ports a production proxy gateway's StreamState logic)
# --------------------------------------------------------------------------- #
def _merge_openai_stream(frames: list[Any]) -> AggregatedStream:
    """Merge a list of OpenAI-style SSE frames into one response body.

    OpenAI-compat streams come in two wire shapes that this merger
    handles transparently:

    - ``/v1/chat/completions`` (``object: chat.completion.chunk``):
      each chunk carries ``choices[].delta`` with ``content`` /
      ``reasoning_content`` / ``tool_calls`` fragments. Concatenate
      string fragments, deep-merge tool-call list items, overwrite
      scalars.
    - ``/v1/completions`` (``object: text_completion``): each chunk
      carries ``choices[].text`` directly (no ``delta`` object), and
      optional per-token ``choices[].logprobs`` whose arrays
      (``tokens``, ``token_logprobs``, ``top_logprobs``,
      ``text_offset``) are length-1 fragments we extend into the full
      record the non-streaming endpoint would have returned. Fireworks'
      ``gpt-oss-*`` family streams this way (see
      ``decoding_sandbox/backends/openai_compat.py``: ``stream_native``
      always hits ``/completions`` when the provider exposes it,
      because that's the only OpenAI endpoint with prompt logprobs).

    ``usage`` blocks on any frame are kept (last write wins). The
    final assembled body is dispatched to ``choices[].text`` shape
    (legacy completions) or ``choices[].message`` shape (chat
    completions) based on the ``object`` field, falling back to the
    presence of ``text`` in any merged choice.
    """
    out = AggregatedStream()
    out.chunks = list(frames)
    merged_choices: dict[int, dict[str, Any]] = {}
    usage_data: dict[str, Any] = {}
    extra_fields: dict[str, Any] = {}

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for k, v in frame.items():
            if k in ("choices", "usage"):
                continue
            if v is not None:
                extra_fields[k] = v
        for choice in frame.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            idx = int(choice.get("index", 0))
            merged = merged_choices.setdefault(idx, {})

            delta = choice.get("delta")
            if isinstance(delta, dict):
                _merge_delta(merged_choices, idx, delta)

            text = choice.get("text")
            if isinstance(text, str):
                merged["text"] = (merged.get("text") or "") + text

            lp = choice.get("logprobs")
            if isinstance(lp, dict):
                existing_lp = merged.setdefault("logprobs", {})
                _merge_logprobs(existing_lp, lp)

            finish = choice.get("finish_reason")
            if finish is not None:
                merged["finish_reason"] = finish
        usage = frame.get("usage")
        if isinstance(usage, dict):
            usage_data = usage

    is_text_completion = str(extra_fields.get("object", "")).lower() == "text_completion" or any(
        isinstance(c, dict) and "text" in c and "content" not in c for c in merged_choices.values()
    )
    if is_text_completion:
        choices_out = _build_text_choices(merged_choices)
        empty_placeholder = {"index": 0, "text": ""}
    else:
        choices_out = _build_chat_choices(merged_choices)
        empty_placeholder = {"index": 0, "message": {"role": "assistant", "content": ""}}
    if not choices_out and frames:
        choices_out.append(empty_placeholder)

    body: dict[str, Any] = dict(extra_fields)
    body["choices"] = choices_out
    if usage_data:
        body["usage"] = usage_data
    out.assembled_body = body

    if usage_data:
        out.prompt_tokens = _coerce_int(usage_data.get("prompt_tokens"))
        out.completion_tokens = _coerce_int(usage_data.get("completion_tokens"))
        out.total_tokens = _coerce_int(usage_data.get("total_tokens"))

    if choices_out:
        first = choices_out[0]
        text_field = first.get("text")
        if isinstance(text_field, str) and text_field:
            out.completion_text = text_field
        else:
            first_message = first.get("message") or {}
            content = first_message.get("content") if isinstance(first_message, dict) else None
            if isinstance(content, str) and content:
                out.completion_text = content
        finish_reason = first.get("finish_reason")
        if isinstance(finish_reason, str):
            out.stop_reason = finish_reason

    model = extra_fields.get("model")
    if isinstance(model, str):
        out.model_resolved = model
    return out


def _build_chat_choices(merged_choices: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Assemble chat-completions-style ``choices[].message`` entries."""
    choices_out: list[dict[str, Any]] = []
    for idx in sorted(merged_choices):
        merged = dict(merged_choices[idx])
        finish = merged.pop("finish_reason", None)
        # Drop legacy fields that don't belong in chat-completion shape.
        merged.pop("text", None)
        merged.pop("logprobs", None)
        message = {k: v for k, v in merged.items() if v is not None}
        message.setdefault("role", "assistant")
        entry: dict[str, Any] = {"index": idx, "message": message}
        if finish is not None:
            entry["finish_reason"] = finish
        choices_out.append(entry)
    return choices_out


def _build_text_choices(merged_choices: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Assemble legacy ``/completions``-style ``choices[].text`` entries.

    The resulting choice dict is shape-compatible with what the
    non-streaming ``/v1/completions`` endpoint returns -- the same
    consumer code path can therefore read either a streamed or
    non-streamed response without branching.
    """
    choices_out: list[dict[str, Any]] = []
    for idx in sorted(merged_choices):
        merged = merged_choices[idx]
        entry: dict[str, Any] = {"index": idx, "text": merged.get("text", "")}
        finish = merged.get("finish_reason")
        if finish is not None:
            entry["finish_reason"] = finish
        lp = merged.get("logprobs")
        if isinstance(lp, dict) and lp:
            entry["logprobs"] = lp
        choices_out.append(entry)
    return choices_out


def _merge_logprobs(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Extend per-chunk legacy-completions logprobs arrays in place.

    Each streamed chunk carries a one-token-wide ``logprobs`` record:

        {"tokens": ["Hel"], "token_logprobs": [-1.2],
         "top_logprobs": [{"Hel": -1.2, "He": -2.1}],
         "text_offset": [42]}

    The non-streaming endpoint returns all of these as parallel arrays
    across the whole completion, so we concatenate the lists here.
    Scalar fields (if any provider ever adds them) take the latest
    value -- matches the OpenAI client library's own behaviour.
    """
    for k, v in source.items():
        if v is None:
            continue
        if isinstance(v, list):
            existing = target.get(k)
            if isinstance(existing, list):
                existing.extend(v)
            else:
                target[k] = list(v)
        else:
            target[k] = v


def _merge_delta(
    merged_choices: dict[int, dict[str, Any]], idx: int, delta: dict[str, Any]
) -> None:
    merged = merged_choices.setdefault(idx, {})
    for key, value in delta.items():
        if value is None:
            continue
        if key in _STRING_MERGE_KEYS and isinstance(value, str):
            merged[key] = (merged.get(key) or "") + value
        elif isinstance(value, list):
            _merge_list_field(merged, key, value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value


def _merge_list_field(merged: dict[str, Any], key: str, items: list[Any]) -> None:
    existing = merged.get(key)
    if not isinstance(existing, list):
        merged[key] = [dict(item) if isinstance(item, dict) else item for item in items]
        return
    for item in items:
        if not isinstance(item, dict):
            existing.append(item)
            continue
        item_idx = item.get("index")
        if item_idx is None:
            existing.append(dict(item))
            continue
        target = None
        for entry in existing:
            if isinstance(entry, dict) and entry.get("index") == item_idx:
                target = entry
                break
        if target is None:
            existing.append(dict(item))
        else:
            _deep_merge_dict(target, item)


def _deep_merge_dict(target: dict[str, Any], source: dict[str, Any]) -> None:
    for k, v in source.items():
        if v is None:
            continue
        existing_v = target.get(k)
        if isinstance(v, dict) and isinstance(existing_v, dict):
            _deep_merge_dict(existing_v, v)
        elif k in _LIST_ITEM_CONCAT_KEYS and isinstance(v, str) and isinstance(existing_v, str):
            target[k] = existing_v + v
        else:
            target[k] = v


# --------------------------------------------------------------------------- #
# dsbx-native merge
# --------------------------------------------------------------------------- #
def _merge_dsbx_stream(frames: list[Any]) -> AggregatedStream:
    """Aggregate dsbx ``/v1/generate/stream`` / ``/v1/spec/stream`` frames.

    The wire emits one event per row (see ``web/streaming.py``):
    ``event=step`` carries one decoded token (we join its ``token_text``
    into the running completion), ``event=usage`` carries provider
    accounting, ``event=done`` is terminal and carries ``stop_reason``
    and an optional ``error``. ``event=round`` (speculative decoding) is
    forwarded verbatim into ``stream_chunks`` -- the UI renders these
    raw because there's no useful "merged" shape for spec rounds.
    """
    out = AggregatedStream()
    out.chunks = list(frames)

    completion_parts: list[str] = []
    spec_completion: str | None = None
    usage_block: dict[str, Any] | None = None
    last_step_event: dict[str, Any] | None = None
    spec_summary: dict[str, Any] | None = None

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        kind = frame.get("event")
        if kind == "step":
            step = frame.get("step") or {}
            decision = step.get("decision") or {}
            token_text = decision.get("token_text")
            if isinstance(token_text, str):
                completion_parts.append(token_text)
            last_step_event = step
        elif kind == "usage":
            # ``usage`` carries the prompt/completion totals directly on
            # the frame (no nested ``usage`` key on the dsbx wire).
            usage_block = {k: v for k, v in frame.items() if k != "event"}
        elif kind == "done":
            sr = frame.get("stop_reason")
            out.stop_reason = sr if isinstance(sr, str) else out.stop_reason
            err = frame.get("error")
            if isinstance(err, str) and err:
                out.error_message = err
            if "completion" in frame and isinstance(frame["completion"], str):
                spec_completion = frame["completion"]
            spec_summary = {k: v for k, v in frame.items() if k != "event"}
        elif kind == "round":
            # Speculative rounds -- nothing to merge, keep as raw chunks.
            pass

    if usage_block is not None:
        out.prompt_tokens = _coerce_int(usage_block.get("prompt_tokens"))
        out.completion_tokens = _coerce_int(usage_block.get("completion_tokens"))
        out.total_tokens = _coerce_int(usage_block.get("total_tokens"))

    if spec_completion is not None:
        out.completion_text = spec_completion
    elif completion_parts:
        out.completion_text = "".join(completion_parts)

    assembled: dict[str, Any] = {
        "events": list(frames),
    }
    if out.completion_text is not None:
        assembled["completion"] = out.completion_text
    if usage_block is not None:
        assembled["usage"] = usage_block
    if out.stop_reason is not None:
        assembled["stop_reason"] = out.stop_reason
    if out.error_message is not None:
        assembled["error"] = out.error_message
    if spec_summary is not None:
        assembled["spec_summary"] = spec_summary
    if last_step_event is not None:
        # The last step's ``step_result.context_text`` carries the
        # context the engine had right before EOS; handy for the UI's
        # "final state" peek.
        sr = last_step_event.get("step_result")
        if isinstance(sr, dict):
            assembled.setdefault("final_step_result", sr)
    out.assembled_body = assembled
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


__all__ = ["AggregatedStream", "aggregate_stream"]
