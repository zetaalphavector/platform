import typer

from zav.agents_sdk.cli.commands.mcp_server_cmd import mcp_server_app
from zav.agents_sdk.cli.commands.platform_cmd import platform_app
from zav.agents_sdk.cli.commands.tools_cmd import tools_app
from zav.agents_sdk.cli.main import app as agents_app
from zav.agents_sdk.version import __version__

za_app = typer.Typer(no_args_is_help=True)

za_app.add_typer(
    agents_app,
    name="agents",
    help="Manage Zeta Alpha agents: create, configure, test, and deploy.",
    no_args_is_help=True,
)
za_app.add_typer(
    tools_app,
    name="tools",
    help="Author custom tools, consumed by both agents and MCP servers.",
    no_args_is_help=True,
)
za_app.add_typer(
    mcp_server_app,
    name="mcp",
    help="Expose your agents' tools as an MCP server: init, serve, dev, install.",
    no_args_is_help=True,
)
za_app.add_typer(
    platform_app,
    name="platform",
    help="Platform credentials and remote operations.",
    no_args_is_help=True,
)


@za_app.command()
def version():
    """
    Prints the current version of the SDK.
    """
    typer.echo(f"Zeta Alpha SDK Version: {__version__}")
