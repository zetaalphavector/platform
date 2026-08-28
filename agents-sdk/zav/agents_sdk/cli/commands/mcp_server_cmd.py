import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated

from zav.agents_sdk.cli.commands.tools_cmd import tools_add
from zav.agents_sdk.cli.project_config import (
    ENV_DIR,
    ProjectConfig,
    ProjectConfigError,
)
from zav.agents_sdk.cli.require import resolve_any_project_dir
from zav.agents_sdk.cli.utils import scaffold_project
from zav.agents_sdk.version import __version__

console = Console()
mcp_server_app = typer.Typer(no_args_is_help=True)

_DEFAULT_IDENTIFIER = "default"
_STARTER_SOURCES = ["web_tools"]
_TOOL_CONFIG_SUFFIX = "_source_configuration"
_LOCAL_PORT = 8000
_REMOTE_URL = "https://api.zeta-alpha.com/v0/service/mcp/"
_INSPECTOR = "@modelcontextprotocol/inspector"


def __configuration_for_sources(sources: List[str]) -> Dict[str, Any]:
    configuration: Dict[str, Any] = {
        "tools_provider_configuration": {"include_sources": sources}
    }
    for source in sources:
        configuration[f"{source}{_TOOL_CONFIG_SUFFIX}"] = {"enabled": True}
    return configuration


def __tool_configuration_from_agent(
    config: ProjectConfig, agent: str
) -> Dict[str, Any]:
    agent_setup = config.get_agent(agent)
    if agent_setup is None:
        console.print(f"  [red]Agent '{agent}' not found[/]")
        raise typer.Exit(code=1)
    agent_configuration = agent_setup.get("agent_configuration", {}) or {}
    # Copy only the tool-relevant blocks — an exposure is a tool bundle, never the
    # agent's LLM, memory, or agent-loop configuration.
    return {
        key: value
        for key, value in agent_configuration.items()
        if key == "tools_provider_configuration" or key.endswith(_TOOL_CONFIG_SUFFIX)
    }


def __seed_configuration(
    config: ProjectConfig, from_agent: Optional[str]
) -> Dict[str, Any]:
    if from_agent:
        return __tool_configuration_from_agent(config, from_agent)
    agents = config.list_agents()
    identifier = agents[0].get("agent_identifier") if agents else None
    if identifier:
        return __tool_configuration_from_agent(config, identifier)
    return __configuration_for_sources(_STARTER_SOURCES)


@mcp_server_app.command("init")
def mcp_init(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(help="Directory of the project to expose over MCP."),
    ] = None,
    from_agent: Annotated[
        Optional[str],
        typer.Option("--from-agent", help="Seed the exposed tools from an agent."),
    ] = None,
):
    """
    Add an MCP server exposure to a project.

    Scaffolds the project skeleton if needed and writes an mcp_setups.json with a
    'default' exposure — seeded from the project's default agent, from --from-agent,
    or a starter tool bundle. Serve it with 'za mcp serve' / 'za mcp dev'.
    """
    if project_dir is None:
        project_dir = typer.prompt("Project directory", default="agents")
    project_dir = os.path.abspath(project_dir)

    scaffold_project(project_dir, __version__)
    config = ProjectConfig(project_dir)

    if config.get_mcp_server(_DEFAULT_IDENTIFIER) is not None:
        console.print(
            f"  [yellow]MCP server '{_DEFAULT_IDENTIFIER}' already exists in "
            f"{project_dir}/[/]"
        )
        raise typer.Exit()

    config.add_mcp_server(
        {
            "mcp_server_identifier": _DEFAULT_IDENTIFIER,
            "configuration": __seed_configuration(config, from_agent),
        }
    )

    relative = os.path.join(
        os.path.basename(project_dir.rstrip("/")), "mcp_setups.json"
    )
    console.print()
    console.print(
        Panel(
            f"[cyan]{relative}[/]  ← MCP exposures",
            title="[bold]✅ MCP server initialized[/]",
            border_style="green",
        )
    )
    console.print("  [bold]Next steps:[/]")
    console.print(f"    cd {project_dir}")
    console.print("    za mcp dev .              [dim]# serve + open the Inspector[/]")
    console.print("    za mcp install            [dim]# wire a host to this server[/]")
    console.print()


@mcp_server_app.command("add")
def mcp_add(
    identifier: Annotated[
        Optional[str],
        typer.Argument(help="Identifier for the MCP server exposure."),
    ] = None,
    from_agent: Annotated[
        Optional[str],
        typer.Option("--from-agent", help="Seed the exposed tools from an agent."),
    ] = None,
    tools: Annotated[
        Optional[str],
        typer.Option("--tools", help="Comma-separated tool sources to expose."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Add an MCP server exposure to mcp_setups.json.
    """
    project_dir = resolve_any_project_dir(project_dir)
    config = ProjectConfig(project_dir)

    if identifier is None:
        identifier = typer.prompt("MCP server identifier", default=_DEFAULT_IDENTIFIER)

    if tools:
        configuration = __configuration_for_sources(
            [t.strip() for t in tools.split(",") if t.strip()]
        )
    else:
        configuration = __seed_configuration(config, from_agent)

    try:
        config.add_mcp_server(
            {"mcp_server_identifier": identifier, "configuration": configuration}
        )
    except ProjectConfigError as error:
        console.print(f"  [red]{error}[/]")
        raise typer.Exit(code=1)

    console.print(f"  [green]✅ MCP server '{identifier}' added[/]")


@mcp_server_app.command("list")
def mcp_list(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    List configured MCP server exposures.
    """
    project_dir = resolve_any_project_dir(project_dir)
    config = ProjectConfig(project_dir)
    setups = config.list_mcp_servers()

    if not setups:
        console.print("  [dim]No MCP servers configured[/]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Identifier", style="cyan")
    table.add_column("Tool sources", style="dim")

    for setup in setups:
        provider = (setup.get("configuration") or {}).get(
            "tools_provider_configuration", {}
        )
        sources = provider.get("include_sources") or "all enabled"
        table.add_row(
            setup.get("mcp_server_identifier", "?"),
            ", ".join(sources) if isinstance(sources, list) else str(sources),
        )

    console.print()
    console.print(table)
    console.print()


@mcp_server_app.command("show")
def mcp_show(
    identifier: Annotated[
        Optional[str],
        typer.Argument(help="MCP server identifier."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Show one MCP server exposure's configuration.
    """
    project_dir = resolve_any_project_dir(project_dir)
    config = ProjectConfig(project_dir)

    if identifier is None:
        identifier = typer.prompt("MCP server identifier", default=_DEFAULT_IDENTIFIER)

    setup = config.get_mcp_server(identifier)
    if setup is None:
        console.print(f"  [red]MCP server '{identifier}' not found[/]")
        raise typer.Exit(code=1)

    console.print()
    console.print(
        Panel(
            json.dumps(setup.get("configuration") or {}, indent=2),
            title=f"[bold]{identifier}[/]",
            border_style="green",
        )
    )
    console.print()


@mcp_server_app.command("remove")
def mcp_remove(
    identifier: Annotated[
        Optional[str],
        typer.Argument(help="MCP server identifier to remove."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Remove an MCP server exposure.
    """
    project_dir = resolve_any_project_dir(project_dir)
    config = ProjectConfig(project_dir)

    if identifier is None:
        identifier = typer.prompt("MCP server identifier to remove")

    if config.get_mcp_server(identifier) is None:
        console.print(f"  [red]MCP server '{identifier}' not found[/]")
        raise typer.Exit(code=1)

    config.remove_mcp_server(identifier)
    console.print(f"  [green]✅ MCP server '{identifier}' removed[/]")


def __prepare_serve_env(project_dir: str) -> str:
    project_dir = os.path.abspath(project_dir)
    os.environ["JSON_LOGGING"] = "0"
    os.environ["ZAV_PROJECT_DIR"] = project_dir
    os.environ["ZAV_AGENT_SETUP_SRC"] = os.path.join(project_dir, "agent_setups.json")
    os.environ["ZAV_SECRET_AGENT_SETUP_SRC"] = os.path.join(
        project_dir, ENV_DIR, "agent_setups.json"
    )
    os.environ["ZAV_MCP_SETUP_SRC"] = os.path.join(project_dir, "mcp_setups.json")
    # The MCP surface is mounted only when serving via `za mcp` — never by
    # `za agents serve`, which stays agents-only.
    os.environ["ZAV_SERVE_MCP"] = "1"
    invocation_dir = os.getcwd()
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [project_dir, invocation_dir, os.getenv("PYTHONPATH")])
    )
    sys.path.insert(0, invocation_dir)
    sys.path.insert(0, project_dir)
    os.chdir(project_dir)
    return project_dir


def __warn_if_public(host: str):
    # The local server has no authentication; the tenant-guarded flow is the
    # hosted server.
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            f"  [yellow]Serving on {host} without authentication — anyone who"
            " can reach this address can call the exposed tools.[/]"
        )


@mcp_server_app.command("serve")
def mcp_serve(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(help="The project directory to serve over MCP."),
    ] = None,
    host: Annotated[
        str,
        typer.Option(help="Host to listen on."),
    ] = "127.0.0.1",
    reload: Annotated[
        bool,
        typer.Option(help="Enable auto-reload."),
    ] = False,
):
    """
    Serve the project's tools over MCP at /mcp (Streamable HTTP).
    """
    project_dir = resolve_any_project_dir(project_dir)
    if not os.path.isfile(os.path.join(project_dir, "mcp_setups.json")):
        console.print("  [red]No mcp_setups.json — run 'za mcp init' first.[/]")
        raise typer.Exit(code=1)
    __warn_if_public(host)
    # __prepare_serve_env chdirs into the project; use the absolute path it
    # returns for anything opened afterwards.
    project_dir = __prepare_serve_env(project_dir)
    with open(os.path.join(project_dir, "mcp_setups.json")) as setups_file:
        setups = json.load(setups_file)
    for setup in setups:
        identifier = setup.get("mcp_server_identifier")
        if identifier:
            console.print(
                f"  [dim]MCP endpoint: http://{host}:{_LOCAL_PORT}/mcp/zetaalpha/"
                f"?mcp_server_identifier={identifier}[/]"
            )
    uvicorn.run(
        "zav.agents_sdk.cli.local_app:app", host=host, port=_LOCAL_PORT, reload=reload
    )


@mcp_server_app.command("dev")
def mcp_dev(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(help="The project directory to serve over MCP."),
    ] = None,
    host: Annotated[
        str,
        typer.Option(help="Host to listen on."),
    ] = "127.0.0.1",
):
    """
    Serve over MCP and open the MCP Inspector for interactive testing.
    """
    project_dir = resolve_any_project_dir(project_dir)
    if not os.path.isfile(os.path.join(project_dir, "mcp_setups.json")):
        console.print("  [red]No mcp_setups.json — run 'za mcp init' first.[/]")
        raise typer.Exit(code=1)

    __warn_if_public(host)
    url = (
        f"http://{host}:{_LOCAL_PORT}/mcp/zetaalpha/"
        f"?mcp_server_identifier={_DEFAULT_IDENTIFIER}"
    )
    if shutil.which("npx"):
        console.print("  [dim]Launching the MCP Inspector (http://localhost:6274)…[/]")
        subprocess.Popen(["npx", "--yes", _INSPECTOR])
        console.print(f"  [dim]Connect it to: {url}[/]")
    else:
        console.print(
            "  [yellow]npx not found — install Node to use the MCP Inspector.[/]"
        )
        console.print(f"  [dim]Then run: npx {_INSPECTOR}  and connect to {url}[/]")

    __prepare_serve_env(project_dir)
    uvicorn.run("zav.agents_sdk.cli.local_app:app", host=host, port=_LOCAL_PORT)


def __mcp_url(
    identifier: str, remote: bool, tenant: Optional[str], host: str, url: Optional[str]
) -> str:
    if url:
        return url
    suffix = f"?mcp_server_identifier={identifier}"
    if remote:
        if not tenant:
            console.print("  [red]--remote requires --tenant.[/]")
            raise typer.Exit(code=1)
        return f"{_REMOTE_URL}{tenant}/{suffix}"
    return f"http://{host}:{_LOCAL_PORT}/mcp/zetaalpha/{suffix}"


def __target_config_path(target: str) -> Optional[Path]:
    home = Path.home()
    return {
        "claude-desktop": home
        / "Library/Application Support/Claude/claude_desktop_config.json",
        "cursor": home / ".cursor/mcp.json",
    }.get(target)


def __write_target(target: str, name: str, entry: Dict[str, Any]) -> bool:
    path = __target_config_path(target)
    if path is None:
        return False
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text() or "{}")
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            existing = loaded
    servers = existing.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
        existing["mcpServers"] = servers
    servers[name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n")
    return True


@mcp_server_app.command("install")
def mcp_install(
    target: Annotated[
        str,
        typer.Argument(help="Where to install: print | claude-desktop | cursor."),
    ] = "print",
    identifier: Annotated[
        str,
        typer.Option("--identifier", help="MCP server identifier to expose."),
    ] = _DEFAULT_IDENTIFIER,
    remote: Annotated[
        bool,
        typer.Option("--remote", help="Point at the hosted Zeta Alpha MCP server."),
    ] = False,
    tenant: Annotated[
        Optional[str],
        typer.Option("--tenant", help="Tenant for the hosted server (with --remote)."),
    ] = None,
    host: Annotated[
        str,
        typer.Option("--host", help="Host of the local server."),
    ] = "127.0.0.1",
    url: Annotated[
        Optional[str],
        typer.Option("--url", help="Override the MCP endpoint URL."),
    ] = None,
):
    """
    Wire a host (Claude Desktop, Cursor) to this MCP server, or print the config.
    """
    name = (
        "zeta-alpha"
        if identifier == _DEFAULT_IDENTIFIER
        else f"zeta-alpha-{identifier}"
    )
    entry = {"type": "http", "url": __mcp_url(identifier, remote, tenant, host, url)}
    block = {"mcpServers": {name: entry}}

    if target != "print" and __write_target(target, name, entry):
        console.print(f"  [green]✅ Installed '{name}' into {target}.[/]")
    else:
        if target != "print":
            console.print(
                f"  [yellow]Unknown target '{target}' — printing config instead.[/]"
            )
        console.print()
        console.print(Panel(json.dumps(block, indent=2), title="[bold]mcpServers[/]"))

    if remote:
        console.print(
            "  [dim]Authenticate with an OIDC bearer token or an X-Auth API key "
            "for the tenant.[/]"
        )


@mcp_server_app.command("add-tool")
def mcp_add_tool(
    name: Annotated[str, typer.Argument(help="Name for the new tool source.")],
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    with_llm: Annotated[
        bool,
        typer.Option(
            "--with-llm",
            help="Scaffold a tool that calls the selected LLM "
            "(model chosen via the model policy).",
        ),
    ] = False,
):
    """Scaffold a custom tool source (alias of `za tools add`)."""
    tools_add(name=name, project_dir=project_dir, with_llm=with_llm)
