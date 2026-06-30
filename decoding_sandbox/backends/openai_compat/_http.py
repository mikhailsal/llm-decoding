"""HTTP transport mixin: retry/backoff and the low-level request helpers."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

from decoding_sandbox.backends.openai_compat._constants import _RETRIABLE_STATUSES
from decoding_sandbox.core import usage as usage_mod

if TYPE_CHECKING:
    import httpx

    from decoding_sandbox.core.config import ProviderConfig

log = logging.getLogger(__name__)


def _parse_retry_after(value: str | None) -> float | None:
    """Return seconds to wait, or ``None`` when the header is missing/unparsable.

    Fireworks (and most OpenAI-compat servers) return ``Retry-After`` as a
    plain integer-seconds string. RFC 7231 also allows an HTTP-date form;
    we don't bother decoding it here -- the caller will fall back to
    exponential backoff in that case, which is just as good.
    """
    if not value:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return max(0.0, float(s))
    except ValueError:
        return None


class _HttpMixin:
    # Composite-class attributes set in ``OpenAICompatBackend.__init__``;
    # declared here under TYPE_CHECKING so mypy sees the surface this
    # mixin reaches into without changing runtime semantics.
    if TYPE_CHECKING:
        provider: ProviderConfig
        model: str
        _client: httpx.Client
        _max_retries: int
        _base_backoff_s: float
        _sleep: Any
        _active_usage: usage_mod.UsageSink | None

        def _provider_flag(self, name: str) -> bool: ...

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if self.provider.require_parameters:
            body.setdefault("provider", {})["require_parameters"] = True
        # Always ask Fireworks for perf_metrics in the response body. The
        # cost is a small object; the benefit is a server-timings panel
        # (TTFT, prefill, generation, speculation acceptance, cached
        # prompt tokens) that turns the educational sandbox into a
        # proper "where is the time going" debugger. Cheap to send; the
        # server ignores the field when unsupported.
        if self._provider_flag("supports_perf_metrics"):
            body.setdefault("perf_metrics_in_response", True)
        # Same justification for ``raw_output``: cheap diagnostics on
        # every Fireworks call. The web layer flushes it via a dedicated
        # SSE frame so consumers that don't care just skip it.
        if self._provider_flag("supports_raw_output"):
            body.setdefault("raw_output", True)
        r = self._request("post", path, json=body)
        data = r.json()
        # If the provider returned a token-usage block (most do for both
        # /completions and /chat/completions), forward it to the active
        # usage sink so the UI can show real provider-side token counts.
        # Falls through silently when no sink is active or no usage was
        # reported, which is the right default for the test/script paths
        # that don't care.
        u = data.get("usage") if isinstance(data, dict) else None
        if isinstance(u, dict):
            usage_mod.record_tokens(
                self._active_usage,
                prompt_tokens=u.get("prompt_tokens"),
                completion_tokens=u.get("completion_tokens"),
                total_tokens=u.get("total_tokens"),
            )
        perf = data.get("perf_metrics") if isinstance(data, dict) else None
        if isinstance(perf, dict):
            usage_mod.record_perf_metrics(self._active_usage, perf)
        raw_out = data.get("raw_output") if isinstance(data, dict) else None
        if isinstance(raw_out, dict):
            usage_mod.record_raw_output(self._active_usage, raw_out)
        return data

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """HTTP call with 429/5xx retry that honors ``Retry-After``.

        This is the single chokepoint for every outbound request: ``_post``
        uses it for JSON bodies, ``_fetch_fireworks_models`` and
        ``fetch_available_models`` use it for the catalogue ``GET``s, and
        the native streaming path uses a sibling :meth:`_stream` so it
        gets the same treatment for the *initial* SSE response code.

        Retry policy:

        - ``2xx`` is returned immediately.
        - ``429`` / ``502`` / ``503`` / ``504`` are retried up to
          ``max_retries`` times. Wait time is the value of the server's
          ``Retry-After`` header (parsed as seconds) when present;
          otherwise exponential backoff (``base_backoff_s * 2**attempt``)
          plus jitter, so concurrent callers don't desynchronize.
        - Every other non-2xx raises via ``raise_for_status`` so the
          caller sees the original ``HTTPStatusError`` (400/401/403/404
          are NEVER retried -- those won't fix themselves and the caller
          needs to see the body for debugging).
        """
        last_response: Any = None
        for attempt in range(self._max_retries + 1):
            # Count every HTTP attempt (not just successful ones) so the
            # usage sink reflects actual pressure on the provider's RPS
            # budget. A 429-then-200 retry shows up as ``requests=2`` --
            # which is the metric the user wants to see when diagnosing
            # rate-limit issues.
            usage_mod.record_request(self._active_usage)
            response = getattr(self._client, method)(path, **kwargs)
            last_response = response
            status = int(getattr(response, "status_code", 0))
            if 200 <= status < 300:
                return response
            if status in _RETRIABLE_STATUSES and attempt < self._max_retries:
                headers = getattr(response, "headers", None) or {}
                wait = _parse_retry_after(
                    headers.get("Retry-After") if hasattr(headers, "get") else None
                )
                if wait is None:
                    # Exponential backoff with jitter. Jitter is small (250
                    # ms) so the user-visible delay tracks the "expected"
                    # schedule, but enough to break ties between racing
                    # callers hitting the same upstream bucket.
                    wait = self._base_backoff_s * (2**attempt) + random.uniform(0.0, 0.25)
                log.warning(
                    "%s: %s %s -> HTTP %d; sleeping %.2fs before retry %d/%d",
                    self.provider.name,
                    method.upper(),
                    path,
                    status,
                    wait,
                    attempt + 1,
                    self._max_retries,
                )
                self._sleep(wait)
                continue
            # Either a non-retriable status, or we've exhausted retries.
            response.raise_for_status()
            return response  # pragma: no cover -- raise_for_status raised
        # Loop exited via retry exhaustion -> raise from the last response.
        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError("unreachable: retry loop completed without a response")
