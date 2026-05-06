from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from zav.agents_sdk.cli.require import require_project, resolve_project_dir

console = Console()
platform_app = typer.Typer(no_args_is_help=True)

ZA_BASE_URL = "https://api.zeta-alpha.com"


@platform_app.command("login")
def platform_login(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option("--base-url", help="Zeta Alpha API base URL."),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", help="Zeta Alpha API key."),
    ] = None,
    tenant: Annotated[
        Optional[str],
        typer.Option("--tenant", help="Tenant name."),
    ] = None,
):
    """
    Set platform credentials for deployment and remote features.
    """
    project_dir = resolve_project_dir(project_dir)
    if base_url is None:
        base_url = typer.prompt("Zeta Alpha API base URL", default=ZA_BASE_URL)
    if api_key is None:
        api_key = typer.prompt("API key", hide_input=True, default="")
    if tenant is None:
        tenant = typer.prompt("Tenant name", default="")

    config = require_project(project_dir)
    config.set_platform_config(base_url=base_url, api_key=api_key, tenant=tenant)
    console.print("  [green]✅ Platform credentials saved[/]")


@platform_app.command("show")
def platform_show(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Show platform credentials (API key masked).
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    platform_cfg = config.get_platform_config()

    if platform_cfg is None:
        console.print("  [dim]No platform credentials configured[/]")
        console.print("  [dim]Run 'za platform login' to set them.[/]")
        return

    base_url = platform_cfg.get("base_url", "Not set")
    api_key = platform_cfg.get("api_key", "")
    tenant = platform_cfg.get("tenant", "")

    if len(api_key) > 9:
        masked = api_key[:9] + "****"
    elif not api_key:
        masked = "Not set"
    else:
        masked = "****"

    console.print()
    console.print(f"  [bold]Base URL:[/] [cyan]{base_url}[/]")
    console.print(f"  [bold]Tenant:[/]   [cyan]{tenant or 'Not set'}[/]")
    console.print(f"  [bold]API Key:[/]  [cyan]{masked}[/]")
    console.print()


@platform_app.command("reset")
def platform_reset(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Reset platform credentials to defaults.
    """
    project_dir = resolve_project_dir(project_dir)

    confirm = typer.confirm("Reset platform credentials to defaults?", default=False)
    if not confirm:
        console.print("  [dim]Cancelled[/]")
        raise typer.Exit()

    config = require_project(project_dir)
    config.set_platform_config(base_url=ZA_BASE_URL, api_key="", tenant="")
    console.print("  [green]✅ Platform credentials reset[/]")
