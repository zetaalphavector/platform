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
    get_source_config_key,
    resolve_alias,
    reverse_alias,
)

console = Console()
source_app = typer.Typer(no_args_is_help=True)


@source_app.command("list")
def source_list(
    provider: Annotated[
        Optional[str],
        typer.Argument(help="Provider name (e.g. tools, skills, memory)."),
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
    List sources for a provider.
    """
    project_dir = resolve_project_dir(project_dir)
    if provider is None:
        provider = typer.prompt("Provider", default="tools")

    prov_key = get_provider_key(provider)
    sources = SourceRegistry.get_sources(provider)
    if not sources:
        console.print(f"  [dim]No configurable sources for '{provider}'[/]")
        return

    config = require_project(project_dir)
    prov_cfg = config.get_provider_config(prov_key, agent) or {}
    include = prov_cfg.get("include_sources", [])
    exclude = prov_cfg.get("exclude_sources", [])

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Internal", style="dim")
    table.add_column("Status")

    for meta in sorted(sources.values(), key=lambda m: m.cli_name):
        internal = meta.source_name
        if include:
            active = internal in include
        elif exclude:
            active = internal not in exclude
        else:
            source_config_key = get_source_config_key(provider, internal)
            agent_cfg = config.get_agent_config(agent)
            source_cfg = agent_cfg.get(source_config_key, {})
            active = source_cfg.get("enabled", False)

        status = "[green]active[/]" if active else "[dim]excluded[/]"
        table.add_row(meta.cli_name, internal, status)

    console.print()
    console.print(table)
    console.print()


@source_app.command("show")
def source_show(
    provider: Annotated[
        Optional[str],
        typer.Argument(help="Provider name."),
    ] = None,
    source: Annotated[
        Optional[str],
        typer.Argument(help="Source name (CLI alias or internal name)."),
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
    Show a source's configuration.
    """
    project_dir = resolve_project_dir(project_dir)
    if provider is None:
        provider = typer.prompt("Provider")
    if source is None:
        source = typer.prompt("Source")

    internal = resolve_alias(provider, source)
    config = require_project(project_dir)
    agent_cfg = config.get_agent_config(agent)

    source_config_key = get_source_config_key(provider, internal)
    source_cfg = agent_cfg.get(source_config_key, {})

    cli_name = reverse_alias(provider, internal)
    console.print()
    console.print(f"  [bold]{cli_name}[/] [dim]({internal})[/]")
    if source_cfg:
        for key, value in source_cfg.items():
            console.print(f"  {key}: {value}")
    else:
        console.print("  [dim]No specific configuration[/]")
    console.print()


@source_app.command("enable")
def source_enable(
    provider: Annotated[
        Optional[str],
        typer.Argument(help="Provider name."),
    ] = None,
    source: Annotated[
        Optional[str],
        typer.Argument(help="Source name."),
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
    Enable a source within a provider.
    """
    project_dir = resolve_project_dir(project_dir)
    if provider is None:
        provider = typer.prompt("Provider")
    if source is None:
        source = typer.prompt("Source")

    internal = resolve_alias(provider, source)
    prov_key = get_provider_key(provider)
    config = require_project(project_dir)
    identifier = agent
    if identifier is None:
        identifier = require_agent(config).get("agent_identifier")

    prov_cfg = config.get_provider_config(prov_key, identifier) or {}
    include = list(prov_cfg.get("include_sources", []))
    exclude = list(prov_cfg.get("exclude_sources", []))

    if internal in exclude:
        exclude.remove(internal)
        prov_cfg["exclude_sources"] = exclude

    if "include_sources" in prov_cfg:
        if internal not in include:
            include.append(internal)
            prov_cfg["include_sources"] = include
    config.set_provider_config(prov_key, prov_cfg, identifier)

    source_config_key = get_source_config_key(provider, internal)
    agent_cfg = config.get_agent_config(identifier)
    source_cfg = agent_cfg.get(source_config_key, {})
    source_cfg["enabled"] = True
    config.set_agent_config(source_config_key, source_cfg, identifier)

    cli_name = reverse_alias(provider, internal)
    console.print(f"  [green]✅ Source '{cli_name}' enabled[/]")


@source_app.command("disable")
def source_disable(
    provider: Annotated[
        Optional[str],
        typer.Argument(help="Provider name."),
    ] = None,
    source: Annotated[
        Optional[str],
        typer.Argument(help="Source name."),
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
    Disable a source within a provider.
    """
    project_dir = resolve_project_dir(project_dir)
    if provider is None:
        provider = typer.prompt("Provider")
    if source is None:
        source = typer.prompt("Source")

    internal = resolve_alias(provider, source)
    prov_key = get_provider_key(provider)
    config = require_project(project_dir)
    identifier = agent
    if identifier is None:
        identifier = require_agent(config).get("agent_identifier")

    prov_cfg = config.get_provider_config(prov_key, identifier) or {}
    exclude = list(prov_cfg.get("exclude_sources", []))

    if internal not in exclude:
        exclude.append(internal)
        prov_cfg["exclude_sources"] = exclude

    include = list(prov_cfg.get("include_sources", []))
    if internal in include:
        include.remove(internal)
        prov_cfg["include_sources"] = include

    config.set_provider_config(prov_key, prov_cfg, identifier)

    source_config_key = get_source_config_key(provider, internal)
    agent_cfg = config.get_agent_config(identifier)
    source_cfg = agent_cfg.get(source_config_key, {})
    source_cfg["enabled"] = False
    config.set_agent_config(source_config_key, source_cfg, identifier)

    cli_name = reverse_alias(provider, internal)
    console.print(f"  [yellow]⚠️  Source '{cli_name}' disabled[/]")
