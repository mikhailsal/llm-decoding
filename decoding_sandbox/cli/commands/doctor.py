"""``dsbx doctor`` -- environment, provider keys, and disk free-space report."""

from __future__ import annotations

import argparse
import os

from rich.table import Table

from decoding_sandbox import __version__
from decoding_sandbox.cli import app
from decoding_sandbox.core import storage
from decoding_sandbox.core.config import Config

# Providers that legitimately do not need an API key (e.g. local OpenAI-compat
# servers). Doctor shows "no key needed" instead of red "missing" for these.
_NO_KEY_PROVIDERS = frozenset({"lmstudio"})


def _mask(secret: str | None, *, no_key_ok: bool = False) -> str:
    if not secret:
        if no_key_ok:
            return "[dim]no key needed[/dim]"
        return "[red]missing[/red]"
    if len(secret) <= 8:
        return "[green]set[/green] (****)"
    return f"[green]set[/green] ({secret[:4]}...{secret[-3:]})"


def cmd_doctor(args: argparse.Namespace, cfg: Config) -> int:
    app.console.rule(f"[bold]Decoding Sandbox doctor[/bold] v{__version__}")

    # Config source / secrets
    app.console.print(f"Config file : [cyan]{cfg.config_path or 'built-in defaults'}[/cyan]")
    app.console.print(f"Secrets file: [cyan]{cfg.secrets_env_file or '(none)'}[/cyan]")
    app.console.print(f"Host        : [cyan]{os.uname().nodename}[/cyan]\n")

    # Provider keys
    ptable = Table(title="Provider API keys", show_edge=False)
    ptable.add_column("provider")
    ptable.add_column("env var")
    ptable.add_column("status")
    ptable.add_column("prompt-logprobs")
    ptable.add_column("max top_logprobs")
    for name, prov in sorted(cfg.providers.items()):
        ptable.add_row(
            name,
            prov.api_key_env,
            _mask(prov.api_key(), no_key_ok=name in _NO_KEY_PROVIDERS),
            "yes" if prov.supports_prompt_logprobs else "no",
            str(prov.max_top_logprobs),
        )
    app.console.print(ptable)

    # Remote dsbx servers (if any configured) -- one probe per [remote.NAME]
    if cfg.remotes:
        _report_remote_servers(cfg)

    # Storage preflight (display all, flag low ones)
    statuses = storage.check_paths(cfg.storage.check_paths, cfg.storage.min_free_gb)
    stable = Table(title=f"Storage (floor {cfg.storage.min_free_gb} GB)", show_edge=False)
    stable.add_column("path")
    stable.add_column("status")
    stable.add_column("free GB", justify="right")
    stable.add_column("total GB", justify="right")
    any_low = False
    for s in statuses:
        if not s.exists:
            stable.add_row(s.path, "[dim]skipped[/dim]", "-", "-")
            continue
        status = "[green]OK[/green]" if s.ok else "[red]LOW[/red]"
        any_low = any_low or not s.ok
        stable.add_row(s.path, status, f"{s.free_gb:.1f}", f"{s.total_gb:.1f}")
    app.console.print(stable)

    # Optional local-engine availability (only meaningful on dsbx-host)
    app.console.print()
    _report_local_engines()

    if any_low:
        app.console.print("\n[red]Warning:[/red] at least one disk is below the free-space floor.")
        return 1
    app.console.print("\n[green]All checks passed.[/green]")
    return 0


def _report_remote_servers(cfg: Config) -> None:
    """Probe each ``[remote.NAME]`` server's ``/v1/info`` and tabulate results.

    A failure to reach a server is rendered red but does not abort
    ``dsbx doctor`` -- the user might be running it on the client
    while ``dsbx-host`` is asleep, and the rest of the report is still
    useful.
    """
    import httpx

    table = Table(title="Remote dsbx servers", show_edge=False)
    table.add_column("name")
    table.add_column("base_url")
    table.add_column("status")
    table.add_column("backend")
    table.add_column("loaded model")
    table.add_column("engine ver", justify="right")

    for name in sorted(cfg.remotes):
        rc = cfg.remotes[name]
        status = "[green]ok[/green]"
        backend = "-"
        model = "-"
        version = "-"
        try:
            with httpx.Client(base_url=rc.base_url, timeout=5.0) as client:
                r = client.get("/v1/info")
                r.raise_for_status()
                info = r.json()
            caps = info.get("capabilities") or {}
            backend = str(info.get("backend_kind", caps.get("name", "?")))
            model = info.get("loaded_model") or "[dim]?[/dim]"
            version = str(info.get("engine_version", "?"))
        except httpx.HTTPError as exc:
            status = f"[red]unreachable[/red] [dim]({type(exc).__name__})[/dim]"
        except Exception as exc:  # noqa: BLE001
            status = f"[red]error[/red] [dim]({type(exc).__name__}: {exc})[/dim]"
        table.add_row(name, rc.base_url, status, backend, model, version)

    app.console.print(table)


def _report_local_engines() -> None:
    table = Table(title="Local engines", show_edge=False)
    table.add_column("component")
    table.add_column("status")

    try:
        import torch  # type: ignore

        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "cpu only"
        table.add_row("torch", f"[green]{torch.__version__}[/green] (cuda={cuda}, {dev})")
    except Exception as exc:  # noqa: BLE001
        table.add_row("torch", f"[yellow]not installed[/yellow] ({type(exc).__name__})")

    try:
        import transformers  # type: ignore

        table.add_row("transformers", f"[green]{transformers.__version__}[/green]")
    except Exception:  # noqa: BLE001
        table.add_row("transformers", "[yellow]not installed[/yellow]")

    try:
        import bitsandbytes  # type: ignore

        table.add_row("bitsandbytes", f"[green]{bitsandbytes.__version__}[/green]")
    except Exception:  # noqa: BLE001
        table.add_row("bitsandbytes", "[yellow]not installed[/yellow]")

    try:
        import llama_cpp  # type: ignore

        ver = getattr(llama_cpp, "__version__", "?")
        # Probing CUDA support without loading a model is hard; just report the
        # binding version. The build flags went through CMAKE_ARGS at install.
        table.add_row("llama-cpp-python", f"[green]{ver}[/green] (for the llamacpp-py backend)")
    except Exception:  # noqa: BLE001
        table.add_row(
            "llama-cpp-python",
            "[yellow]not installed[/yellow] (needed for llamacpp-py)",
        )

    app.console.print(table)
