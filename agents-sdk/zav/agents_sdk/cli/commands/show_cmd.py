from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated

from zav.agents_sdk.cli.require import (
    require_agent,
    require_project,
    resolve_project_dir,
)
from zav.agents_sdk.cli.source_aliases import (
    SourceRegistry,
    get_providers,
    get_source_config_key,
)

console = Console()


def __show_project_overview(project_dir: str, config, agent_setup: dict):
    identifier = agent_setup.get("agent_identifier", "?")
    agent_name = agent_setup.get("agent_name", "?")

    llm = config.resolve_agent_llm(agent_setup.get("agent_identifier"))
    vendor = llm.get("vendor", "?")
    model_name = llm.get("model_configuration", {}).get("name", "?")

    agent_cfg = agent_setup.get("agent_configuration", {})

    is_builtin = agent_name == "agent"
    agent_type = (
        "[green]built-in[/]" if is_builtin else f"[yellow]custom ({agent_name})[/]"
    )

    console.print()
    info_table = Table.grid(padding=(0, 2))
    info_table.add_row("[bold]Project[/]", f"[cyan]{project_dir}[/]")
    info_table.add_row("[bold]Agent[/]", f"[cyan]{identifier}[/] {agent_type}")
    info_table.add_row("[bold]Model[/]", f"[cyan]{model_name}[/] [dim]({vendor})[/]")
    console.print(Panel(info_table, border_style="blue"))

    provider_table = Table(title="Capabilities", show_header=True, header_style="bold")
    provider_table.add_column("Provider", style="cyan")
    provider_table.add_column("Status")
    provider_table.add_column("Sources / Details", style="dim")

    for prov_name, prov_info in get_providers().items():
        display = prov_info.display_name
        prov_cfg = agent_cfg.get(prov_info.config_key, {})
        enabled = prov_cfg.get("enabled", False) if prov_cfg else False
        status = "[green]enabled[/]" if enabled else "[dim]disabled[/]"

        source_count = ""
        sources = SourceRegistry.get_sources(prov_name)
        if sources and enabled:
            include = prov_cfg.get("include_sources", [])
            exclude = prov_cfg.get("exclude_sources", [])
            total = len(sources)
            if include:
                active = len(include)
            elif exclude:
                active = total - len(exclude)
            else:
                active = sum(
                    1
                    for meta in sources.values()
                    if agent_cfg.get(
                        get_source_config_key(prov_name, meta.source_name), {}
                    ).get("enabled", False)
                )
            source_count = f"{active}/{total} {prov_info.cli_name}"
        elif prov_name == "mcp" and enabled:
            server_count = len(prov_cfg.get("servers", []))
            source_count = f"{server_count} {prov_info.cli_name}"

        provider_table.add_row(display, status, source_count)

    console.print(provider_table)
    console.print()

    skills = config.list_skills()
    instructions = config.list_instructions()
    specs = config.list_specs()

    files_table = Table.grid(padding=(0, 2))
    files_table.add_row(
        "[bold]Skills[/]",
        f"[cyan]{len(skills)}[/] files in skills/",
    )
    files_table.add_row(
        "[bold]Instructions[/]",
        f"[cyan]{len(instructions)}[/] files in instructions/",
    )
    files_table.add_row(
        "[bold]Specs[/]",
        f"[cyan]{len(specs)}[/] files in specs/",
    )
    console.print(files_table)
    console.print()


def __show_agent_detail(config, agent_setup: dict):
    identifier = agent_setup.get("agent_identifier", "?")
    agent_name = agent_setup.get("agent_name", "?")
    llm = config.resolve_agent_llm(identifier)
    agent_cfg = agent_setup.get("agent_configuration", {})

    is_builtin = agent_name == "agent"
    agent_type = (
        "[green]built-in[/]" if is_builtin else f"[yellow]custom ({agent_name})[/]"
    )
    vendor = llm.get("vendor", "?")
    model_name = llm.get("model_configuration", {}).get("name", "?")

    console.print()
    info_table = Table.grid(padding=(0, 2))
    info_table.add_row("[bold]Identifier[/]", f"[cyan]{identifier}[/]")
    info_table.add_row("[bold]Type[/]", agent_type)
    info_table.add_row("[bold]Model[/]", f"[cyan]{model_name}[/] [dim]({vendor})[/]")
    console.print(
        Panel(info_table, title=f"[bold]{identifier}[/]", border_style="blue")
    )

    provider_table = Table(title="Capabilities", show_header=True, header_style="bold")
    provider_table.add_column("Provider", style="cyan")
    provider_table.add_column("Status")
    provider_table.add_column("Sources / Details", style="dim")

    for prov_name, prov_info in get_providers().items():
        display = prov_info.display_name
        prov_cfg = agent_cfg.get(prov_info.config_key, {})
        enabled = prov_cfg.get("enabled", False) if prov_cfg else False
        status = "[green]enabled[/]" if enabled else "[dim]disabled[/]"

        source_count = ""
        sources = SourceRegistry.get_sources(prov_name)
        if sources and enabled:
            include = prov_cfg.get("include_sources", [])
            exclude = prov_cfg.get("exclude_sources", [])
            total = len(sources)
            if include:
                active = len(include)
            elif exclude:
                active = total - len(exclude)
            else:
                active = sum(
                    1
                    for meta in sources.values()
                    if agent_cfg.get(
                        get_source_config_key(prov_name, meta.source_name), {}
                    ).get("enabled", False)
                )
            source_count = f"{active}/{total} {prov_info.cli_name}"
        elif prov_name == "mcp" and enabled:
            server_count = len(prov_cfg.get("servers", []))
            source_count = f"{server_count} {prov_info.cli_name}"

        provider_table.add_row(display, status, source_count)

    console.print(provider_table)
    console.print()


def show_command(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Agent identifier. Omit for project overview."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Show project overview or a specific agent's configuration.

    Without arguments, shows the project overview: model, capabilities,
    skills, instructions, and specs. With an agent name, shows that agent's
    detailed configuration.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)

    if name is None:
        __show_project_overview(project_dir, config, require_agent(config))
    else:
        agent_setup = config.get_agent(name)
        if agent_setup is None:
            console.print(f"  [red]Agent '{name}' not found[/]")
            raise typer.Exit(code=1)
        __show_agent_detail(config, agent_setup)
