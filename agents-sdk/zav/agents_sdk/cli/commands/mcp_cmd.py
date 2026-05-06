import json
from typing import Optional

import typer
from pydantic_core import PydanticUndefined
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from zav.agents_sdk.cli.project_config import ProjectConfig
from zav.agents_sdk.cli.require import (
    require_agent,
    require_project,
    resolve_project_dir,
)

console = Console()
mcp_app = typer.Typer(no_args_is_help=True)

MCP_PROVIDER_KEY = "mcp_tools_provider_configuration"


def __get_mcp_config(config: ProjectConfig, agent: Optional[str] = None) -> dict:
    agent_cfg = config.get_agent_config(agent)
    return agent_cfg.get(MCP_PROVIDER_KEY, {})


def __get_servers(config: ProjectConfig, agent: Optional[str] = None) -> list:
    mcp_cfg = __get_mcp_config(config, agent)
    return mcp_cfg.get("servers", [])


@mcp_app.command("add")
def mcp_add(
    name: Annotated[
        Optional[str],
        typer.Argument(help="MCP server name."),
    ] = None,
    transport: Annotated[
        str,
        typer.Option("--transport", "-t", help="Transport type (stdio/sse/http)."),
    ] = "stdio",
    url: Annotated[
        Optional[str],
        typer.Option("--url", help="Server URL (for sse/http transport)."),
    ] = None,
    command: Annotated[
        Optional[str],
        typer.Option("--command", help="Command to run (for stdio transport)."),
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
    Add an MCP server configuration.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("MCP server name")

    config = require_project(project_dir)
    identifier = agent
    if identifier is None:
        identifier = require_agent(config).get("agent_identifier")

    servers = __get_servers(config, identifier)
    for s in servers:
        if s.get("name") == name:
            console.print(f"  [yellow]MCP server '{name}' already exists[/]")
            raise typer.Exit(code=1)

    server_entry: dict = {"name": name, "transport": transport}
    if transport == "stdio":
        if command is None:
            command = typer.prompt("Command to run")
        server_entry["command"] = command
    else:
        if url is None:
            url = typer.prompt("Server URL")
        server_entry["url"] = url

    servers.append(server_entry)
    mcp_cfg = __get_mcp_config(config, identifier)
    mcp_cfg["enabled"] = True
    mcp_cfg["servers"] = servers
    config.set_agent_config(MCP_PROVIDER_KEY, mcp_cfg, identifier)

    console.print(f"  [green]✅ MCP server '{name}' added ({transport})[/]")


@mcp_app.command("list")
def mcp_list(
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
    List configured MCP servers.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    servers = __get_servers(config, agent)

    if not servers:
        console.print("  [dim]No MCP servers configured[/]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Transport")
    table.add_column("URL/Command", style="dim")

    for s in servers:
        endpoint = s.get("url", s.get("command", ""))
        table.add_row(s.get("name", "?"), s.get("transport", "?"), str(endpoint))

    console.print()
    console.print(table)
    console.print()


@mcp_app.command("show")
def mcp_show(
    name: Annotated[
        Optional[str],
        typer.Argument(help="MCP server name."),
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
    Show an MCP server's configuration.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("MCP server name")

    config = require_project(project_dir)
    servers = __get_servers(config, agent)

    for s in servers:
        if s.get("name") == name:
            console.print()
            console.print(f"  [bold]{name}[/]")
            for key, value in s.items():
                if key != "name":
                    console.print(f"  {key}: {value}")
            console.print()
            return

    console.print(f"  [red]MCP server '{name}' not found[/]")


@mcp_app.command("remove")
def mcp_remove(
    name: Annotated[
        Optional[str],
        typer.Argument(help="MCP server name."),
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
    Remove an MCP server configuration.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("MCP server name to remove")

    config = require_project(project_dir)
    identifier = agent
    if identifier is None:
        identifier = require_agent(config).get("agent_identifier")

    servers = __get_servers(config, identifier)
    new_servers = [s for s in servers if s.get("name") != name]
    if len(new_servers) == len(servers):
        console.print(f"  [red]MCP server '{name}' not found[/]")
        raise typer.Exit(code=1)

    confirm = typer.confirm(f"Remove MCP server '{name}'?", default=False)
    if not confirm:
        console.print("  [dim]Cancelled[/]")
        raise typer.Exit()

    mcp_cfg = __get_mcp_config(config, identifier)
    mcp_cfg["servers"] = new_servers
    config.set_agent_config(MCP_PROVIDER_KEY, mcp_cfg, identifier)
    console.print(f"  [green]✅ MCP server '{name}' removed[/]")


SETTINGS_SKIP_FIELDS = frozenset({"enabled", "servers", "tool_streaming"})


@mcp_app.command("settings")
def mcp_settings(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    agent: Annotated[
        Optional[str],
        typer.Option("--agent", help="Agent identifier."),
    ] = None,
):
    """Interactively configure MCP provider-level settings."""
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    identifier = agent
    if identifier is None:
        identifier = require_agent(config).get("agent_identifier")

    mcp_cfg = dict(__get_mcp_config(config, identifier))

    console.print()
    console.print("  [bold]MCP Provider Settings[/]")
    console.print(
        "  [dim]Press Enter to keep current/default. Type 'null' to clear.[/]"
    )
    console.print()

    is_enabled = mcp_cfg.get("enabled", False)
    enable_choice = typer.confirm(
        "  Enable MCP Tools provider?",
        default=is_enabled,
    )
    changed = False
    if enable_choice != is_enabled:
        mcp_cfg["enabled"] = enable_choice
        changed = True

    if not enable_choice:
        if changed:
            config.set_agent_config(MCP_PROVIDER_KEY, mcp_cfg, identifier)
            console.print()
            console.print("  [green]✅ MCP Tools provider disabled[/]")
        else:
            console.print()
            console.print("  [dim]No changes made[/]")
        return

    # Deferred import: adapter modules pull in heavy SDK internals — importing at
    # module level adds seconds to CLI startup for commands that don't need them.
    from zav.agents_sdk.adapters.mcp.tools_provider import MCPToolsProviderConfiguration

    for name, field_info in MCPToolsProviderConfiguration.model_fields.items():
        if name in SETTINGS_SKIP_FIELDS:
            continue

        current = mcp_cfg.get(name)
        default = field_info.default
        if default is PydanticUndefined:
            default = (
                field_info.default_factory() if field_info.default_factory else None
            )
        display_current = current if current is not None else default
        hint = f" [{display_current}]" if display_current is not None else ""

        desc = field_info.description
        if desc:
            console.print(f"  [dim]{desc}[/]")

        raw = typer.prompt(
            f"  {name}{hint}",
            default="",
            show_default=False,
        )

        if raw == "":
            continue

        if raw.lower() == "null":
            if name in mcp_cfg:
                del mcp_cfg[name]
                changed = True
            continue

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = raw

        mcp_cfg[name] = parsed
        changed = True

    if changed:
        config.set_agent_config(MCP_PROVIDER_KEY, mcp_cfg, identifier)
        console.print()
        console.print("  [green]✅ MCP provider settings updated[/]")
    else:
        console.print()
        console.print("  [dim]No changes made[/]")
