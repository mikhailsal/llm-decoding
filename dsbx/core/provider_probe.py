"""Live capability probe for logprob-capable providers.

This freezes the manual curl tests from the planning phase into a repeatable
check. For each configured provider it makes a tiny real request and reports:

- chat logprobs: does /chat/completions return per-token top_logprobs?
- prompt logprobs: does /completions with echo=true score the whole prompt?
  (only attempted where the provider advertises support, e.g. Fireworks)

Findings recorded during planning (June 2026), to be reconfirmed by this tool:
- Fireworks  : chat top_logprobs<=5 AND whole-context via echo. Works on top models.
- NVIDIA NIM : chat top_logprobs<=20. No /completions (404) -> no prompt logprobs.
- OpenRouter : chat logprobs only with provider.require_parameters=true.
- Gemini AI Studio: logprobs disabled (capability gate) -> not configured here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from dsbx.core.config import Config, ProviderConfig

PROMPT = "The capital of France is"


@dataclass
class ProbeResult:
    provider: str
    model: str
    chat_logprobs: str  # "ok (N alts)" | "no" | "err: ..."
    prompt_logprobs: str  # "ok (N tokens)" | "n/a" | "no" | "err: ..."


def _headers(prov: ProviderConfig, key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _probe_chat(prov: ProviderConfig, key: str, model: str) -> str:
    top = min(5, prov.max_top_logprobs)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 2,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top,
    }
    if prov.require_parameters:
        body["provider"] = {"require_parameters": True}
    try:
        resp = httpx.post(
            f"{prov.base_url}/chat/completions",
            headers=_headers(prov, key),
            json=body,
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        return f"err: {type(exc).__name__}"
    if resp.status_code != 200:
        return f"err: HTTP {resp.status_code}"
    try:
        data = resp.json()
    except ValueError:
        return "err: non-JSON response"
    choices = data.get("choices") or [{}]
    lp = choices[0].get("logprobs")
    content = (lp or {}).get("content") if isinstance(lp, dict) else None
    if content:
        n_alts = len(content[0].get("top_logprobs", []))
        return f"ok ({n_alts} alts)"
    return "no logprobs field"


def _probe_prompt(prov: ProviderConfig, key: str, model: str) -> str:
    if not prov.supports_prompt_logprobs:
        return "n/a"
    body = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 3,
        "echo": True,
    }
    try:
        resp = httpx.post(
            f"{prov.base_url}/completions",
            headers=_headers(prov, key),
            json=body,
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        return f"err: {type(exc).__name__}"
    if resp.status_code == 404:
        return "no (/completions 404)"
    if resp.status_code != 200:
        return f"err: HTTP {resp.status_code}"
    try:
        data = resp.json()
    except ValueError:
        return "err: non-JSON response"
    choices = data.get("choices") or [{}]
    lp = choices[0].get("logprobs") or {}
    tokens = lp.get("tokens")
    if tokens:
        return f"ok ({len(tokens)} tokens)"
    return "no prompt logprobs"


def probe_provider(prov: ProviderConfig, model: str | None) -> ProbeResult:
    key = prov.api_key()
    use_model = model or prov.default_model
    if not key:
        return ProbeResult(prov.name, use_model, "err: no API key", "err: no API key")
    return ProbeResult(
        provider=prov.name,
        model=use_model,
        chat_logprobs=_probe_chat(prov, key, use_model),
        prompt_logprobs=_probe_prompt(prov, key, use_model),
    )


def run_probe(
    cfg: Config,
    providers: list[str] | None = None,
    model: str | None = None,
    console: Any = None,
) -> int:
    names = providers or sorted(cfg.providers)
    results: list[ProbeResult] = []
    for name in names:
        try:
            prov = cfg.provider(name)
        except KeyError as exc:
            if console:
                console.print(f"[red]{exc}[/red]")
            return 2
        if console:
            console.print(f"[dim]probing {name} ({model or prov.default_model})...[/dim]")
        results.append(probe_provider(prov, model))

    if console is not None:
        from rich.table import Table

        table = Table(title="Provider logprob capabilities (live)")
        table.add_column("provider")
        table.add_column("model")
        table.add_column("chat logprobs")
        table.add_column("prompt logprobs (whole context)")
        for r in results:
            table.add_row(r.provider, r.model, r.chat_logprobs, r.prompt_logprobs)
        console.print(table)
    else:
        for r in results:
            print(f"{r.provider}\t{r.model}\t{r.chat_logprobs}\t{r.prompt_logprobs}")

    any_err = any(r.chat_logprobs.startswith("err") for r in results)
    return 1 if any_err else 0
