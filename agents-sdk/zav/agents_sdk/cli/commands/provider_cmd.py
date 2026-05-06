from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from zav.agents_sdk.cli.require import (
    require_agent,
    require_project,
    resolve_project_dir,
)
from zav.agents_sdk.cli.source_aliases import (
    SourceRegistry,
    get_provider_key,
    get_providers,
)

console = Console()
provider_app = typer.Typer(no_args_is_help=True)


@provider_app.command("list")
def provider_list(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    agent: Annotated[
        Optional[str],
        typer.Option("--agent", help="Agent identifier."),
    ] = None,
):
    """
    List all providers and their status.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    agent_setup = require_agent(config, agent)
    agent_cfg = agent_setup.get("agent_configuration", {})

    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Sources", style="dim")

    for prov_name, prov_info in get_providers().items():
        display = prov_info.display_name
        prov_cfg = agent_cfg.get(prov_info.config_key, {})
        enabled = prov_cfg.get("enabled", False) if prov_cfg else False
        status = "[green]enabled[/]" if enabled else "[dim]disabled[/]"

        sources = SourceRegistry.get_sources(prov_name)
        source_info = f"{len(sources)} available" if sources else ""

        table.add_row(display, status, source_info)

    console.print()
    console.print(table)
    console.print()


@provider_app.command("show")
def provider_show(
    provider: Annotated[
        Optional[str],
        typer.Argument(help="Provider name."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    agent: Annotated[
        Optional[str],
        typer.Option("--agent", help="Agent identifier."),
    ] = None,
):
    """
    Show a provider's configuration.
    """
    project_dir = resolve_project_dir(project_dir)
    if provider is None:
        provider = typer.prompt("Provider", default="tools")

    prov_key = get_provider_key(provider)
    config = require_project(project_dir)
    prov_cfg = config.get_provider_config(prov_key, agent if agent else None)

    prov_info = get_providers().get(provider, None)
    display = prov_info.display_name if prov_info else provider
    console.print()
    console.print(f"  [bold]{display} Provider[/]")

    if prov_cfg is None:
        console.print("  [dim]Not configured[/]")
    else:
        enabled = prov_cfg.get("enabled", False)
        status = "[green]enabled[/]" if enabled else "[dim]disabled[/]"
        console.print(f"  Status: {status}")

        include = prov_cfg.get("include_sources", [])
        exclude = prov_cfg.get("exclude_sources", [])
        if include:
            console.print(f"  Include: {', '.join(include)}")
        if exclude:
            console.print(f"  Exclude: {', '.join(exclude)}")

        for key, value in prov_cfg.items():
            if key in ("enabled", "include_sources", "exclude_sources"):
                continue
            console.print(f"  {key}: {value}")
    console.print()


@provider_app.command("enable")
def provider_enable(
    provider: Annotated[
        Optional[str],
        typer.Argument(help="Provider name."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    agent: Annotated[
        Optional[str],
        typer.Option("--agent", help="Agent identifier."),
    ] = None,
):
    """
    Enable a provider.
    """
    project_dir = resolve_project_dir(project_dir)
    if provider is None:
        provider = typer.prompt("Provider to enable")

    prov_key = get_provider_key(provider)
    config = require_project(project_dir)
    identifier = agent
    if identifier is None:
        identifier = require_agent(config).get("agent_identifier")

    existing = config.get_provider_config(prov_key, identifier) or {}
    existing["enabled"] = True
    config.set_provider_config(prov_key, existing, identifier)

    prov_info = get_providers().get(provider, None)
    display = prov_info.display_name if prov_info else provider
    console.print(f"  [green]✅ {display} provider enabled[/]")


@provider_app.command("disable")
def provider_disable(
    provider: Annotated[
        Optional[str],
        typer.Argument(help="Provider name."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    agent: Annotated[
        Optional[str],
        typer.Option("--agent", help="Agent identifier."),
    ] = None,
):
    """
    Disable a provider.
    """
    project_dir = resolve_project_dir(project_dir)
    if provider is None:
        provider = typer.prompt("Provider to disable")

    prov_key = get_provider_key(provider)
    config = require_project(project_dir)
    identifier = agent
    if identifier is None:
        identifier = require_agent(config).get("agent_identifier")

    existing = config.get_provider_config(prov_key, identifier) or {}
    existing["enabled"] = False
    config.set_provider_config(prov_key, existing, identifier)

    prov_info = get_providers().get(provider, None)
    display = prov_info.display_name if prov_info else provider
    console.print(f"  [yellow]⚠️  {display} provider disabled[/]")
