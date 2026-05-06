import base64
import io
import json
import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from zav.agents_sdk.cli.project_config import ProjectConfig
from zav.agents_sdk.cli.require import require_project, resolve_project_dir

console = Console()
deploy_app = typer.Typer(no_args_is_help=True)


def __get_project_directory(project_dir: Optional[str] = None) -> str:
    if project_dir is None:
        project_dir = os.getcwd()
        if not ProjectConfig.is_valid_project(project_dir):
            project_dir = typer.prompt("Enter the project directory", default="agents")
    if not ProjectConfig.is_valid_project(project_dir):
        typer.echo(
            "Invalid project directory. Please ensure you are in a valid project "
            "directory."
        )
        typer.echo(
            typer.style(
                "You can create a new project directory by running 'za agents init'.",
                fg=typer.colors.GREEN,
            )
        )
        raise typer.Exit()
    return project_dir  # type: ignore


def __create_agent_code_bundle(project_dir: str, project_name: str):
    # Deferred import: domain modules pull in heavy SDK internals — importing at
    # module level adds seconds to CLI startup for commands that don't need them.
    from zav.agents_sdk.domain.agent_code_bundle import AgentCodeBundle

    existing_pythonpath = os.getenv("PYTHONPATH")
    current_path = os.getcwd()
    os.environ["PYTHONPATH"] = (
        f"{existing_pythonpath}:{current_path}" if existing_pythonpath else current_path
    )

    AgentCodeBundle.load_agent_registries_from(project_dir=project_dir)

    config = ProjectConfig(project_dir)
    agent_names = [
        a.get("agent_identifier", a.get("agent_name", "agent"))
        for a in config.list_agents()
    ]

    agent_code_bundle = AgentCodeBundle.from_project_dir(
        project=project_name, agent_names=agent_names, project_dir=project_dir
    )
    return agent_code_bundle


def __load_client_config(project_dir: str):
    config_path = os.path.join(project_dir, "env", "zav_config.json")
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def __store_client_config(project_dir: str, base_url: str, api_key: str, tenant: str):
    config_path = os.path.join(project_dir, "env", "zav_config.json")
    config = {"base_url": base_url, "api_key": api_key, "tenant": tenant}
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as file:
        json.dump(config, file, indent=4)
    return config_path


_EXCLUDED_BOT_CONFIG_KEYS = {"base_url", "tenant"}


def __build_agent_setup(
    project: str, agent_name: str, agent_configuration: dict
) -> dict:
    bot_configuration = {
        k: v
        for k, v in agent_configuration.items()
        if k not in _EXCLUDED_BOT_CONFIG_KEYS
    }
    return {
        "bot_identifier": f"{project}:{agent_name}",
        "agent_name": f"{project}:{agent_name}",
        "bot_configuration": bot_configuration,
    }


@deploy_app.command("list")
def deploy_list(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    List deployed projects on the Zeta Alpha Platform.
    """
    from zav.chat_service import ApiClient, Configuration
    from zav.chat_service.apis import AgentBundlesApi

    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    platform_cfg = config.get_platform_config()

    if platform_cfg is None:
        console.print("  [red]No platform credentials configured[/]")
        console.print("  [dim]Run 'za platform login' first.[/]")
        raise typer.Exit(code=1)

    host = platform_cfg["base_url"]
    if not host.rstrip("/").endswith("v0/service"):
        host = host.rstrip("/") + "/v0/service"

    api_config = Configuration(host=host)
    api_client = ApiClient(api_config)
    api_client.set_default_header("X-Auth", platform_cfg["api_key"])
    agent_bundles_api = AgentBundlesApi(api_client)

    available = agent_bundles_api.filter_agent_bundle(
        user_roles="",
        user_tenants="",
        requester_uuid="",
        tenant=platform_cfg["tenant"],
    )

    if available.count == 0:
        console.print("  [dim]No projects found[/]")
        return

    console.print()
    console.print(
        f"  [bold]Deployed projects for tenant [cyan]{platform_cfg['tenant']}[/][/]"
    )
    for bundle in available.results:
        for agent_name in bundle.agent_names:
            created = bundle.created_at.strftime("%b %d, %Y %H:%M")
            updated = bundle.last_updated_at.strftime("%b %d, %Y %H:%M")
            console.print(
                f"  [magenta]• {bundle.project}:{agent_name}[/] "
                f"[dim]Created: {created} | Updated: {updated}[/]"
            )
    console.print()


@deploy_app.command("delete")
def deploy_delete(
    project_name: Annotated[
        Optional[str],
        typer.Argument(help="The project name to delete."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Delete a deployed project from the Zeta Alpha Platform.
    """
    from zav.chat_service import ApiClient, Configuration
    from zav.chat_service.apis import AgentBundlesApi
    from zav.chat_service.exceptions import NotFoundException

    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    platform_cfg = config.get_platform_config()

    if platform_cfg is None:
        console.print("  [red]No platform credentials configured[/]")
        console.print("  [dim]Run 'za platform login' first.[/]")
        raise typer.Exit(code=1)

    if project_name is None:
        project_name = os.path.basename(os.path.abspath(project_dir))

    if ":" in project_name:
        project_name = project_name.split(":")[0]

    host = platform_cfg["base_url"]
    if not host.rstrip("/").endswith("v0/service"):
        host = host.rstrip("/") + "/v0/service"

    api_config = Configuration(host=host)
    api_client = ApiClient(api_config)
    api_client.set_default_header("X-Auth", platform_cfg["api_key"])
    agent_bundles_api = AgentBundlesApi(api_client)

    try:
        agent_bundles_api.retrieve_agent_bundle(
            user_roles="",
            user_tenants="",
            requester_uuid="",
            project=project_name,
            tenant=platform_cfg["tenant"],
        )
    except NotFoundException:
        console.print(
            f"  [red]Project '{project_name}' not found on tenant "
            f"'{platform_cfg['tenant']}'[/]"
        )
        raise typer.Exit(code=1)

    confirm = typer.confirm(
        f"Delete project '{project_name}' from tenant '{platform_cfg['tenant']}'?",
        default=False,
    )
    if not confirm:
        console.print("  [dim]Cancelled.[/]")
        raise typer.Exit()

    agent_bundles_api.delete_agent_bundle(
        user_roles="",
        user_tenants="",
        requester_uuid="",
        project=project_name,
        tenant=platform_cfg["tenant"],
    )

    console.print(f"  [green]✅ Project '{project_name}' deleted.[/]")


@deploy_app.command("bundle")
def deploy_bundle(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(
            callback=__get_project_directory,
            help="The project directory where the agents are located.",
        ),
    ] = None,
    project_name: Annotated[
        Optional[str],
        typer.Option(
            help="The name of the project under which the agents will be bundled.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            help="The directory where the bundle will be stored. If not provided, the "
            "bundle will be stored in the project directory.",
        ),
    ] = None,
):
    """
    Creates a bundle of the current project.

    Runs validation first. If there are errors, the bundle is not created.
    If there are only warnings, you will be prompted to continue.
    """
    from zav.agents_sdk.cli.commands.validate_cmd import run_validation

    assert project_dir is not None

    validation_errors, validation_warnings = run_validation(project_dir)
    if validation_errors > 0:
        console.print("  [red]Fix errors before bundling.[/]")
        raise typer.Exit(code=1)
    if validation_warnings > 0:
        if not typer.confirm("Continue with warnings?", default=True):
            raise typer.Exit()

    if project_name is None:
        project_name = os.path.basename(os.path.abspath(project_dir))

    agent_code_bundle = __create_agent_code_bundle(
        project_dir=project_dir, project_name=project_name
    )
    if output_dir is None:
        output_dir = project_dir

    build_dir = os.path.join(output_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    bundle_path = os.path.join(build_dir, "build.zip")
    with open(bundle_path, "wb") as file:
        file.write(agent_code_bundle.agent_bundle)

    project_config = ProjectConfig(project_dir)
    for agent_name in agent_code_bundle.agent_names:
        agent_setup = project_config.get_agent(agent_name)
        agent_configuration = (
            agent_setup.get("agent_configuration", {}) if agent_setup else {}
        )
        bot_config_data = __build_agent_setup(
            agent_code_bundle.project, agent_name, agent_configuration
        )
        config_path = os.path.join(build_dir, f"{agent_name}.json")
        with open(config_path, "w") as f:
            json.dump(bot_config_data, f, indent=2)
            f.write("\n")
        console.print(f"  [dim]Agent setup: {config_path}[/]")

    console.print(f"  [green]✅ Bundle created: {bundle_path}[/]")
    console.print()
    console.print(
        "  [dim]Next: run 'za agents deploy upload' to deploy to the platform[/]"
    )


@deploy_app.command("upload")
def deploy_upload(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(
            callback=__get_project_directory,
            help="The project directory where the agents are located.",
        ),
    ] = None,
    project_name: Annotated[
        Optional[str],
        typer.Option(
            help="The name of the project under which the agents will be uploaded.",
        ),
    ] = None,
):
    """
    Uploads the current project to the Zeta Alpha Platform.
    """
    from zav.chat_service import ApiClient, Configuration
    from zav.chat_service.apis import AgentBundlesApi
    from zav.chat_service.exceptions import NotFoundException
    from zav.chat_service.models import AgentBundleForm, AgentBundlePatch

    assert project_dir is not None

    if project_name is None:
        project_name = os.path.basename(os.path.abspath(project_dir))

    try:
        config = __load_client_config(project_dir)
    except FileNotFoundError:
        base_url = typer.prompt(
            "Enter the Zeta Alpha API base URL",
            default="https://api.zeta-alpha.com",
        )
        api_key = typer.prompt(
            "Enter your Zeta Alpha API key",
            default="",
            hide_input=True,
        )
        tenant = typer.prompt("Enter your Zeta Alpha tenant", default="")
        __store_client_config(
            project_dir=project_dir,
            base_url=base_url,
            api_key=api_key,
            tenant=tenant,
        )
        config = __load_client_config(project_dir)

    agent_code_bundle = __create_agent_code_bundle(
        project_dir=project_dir, project_name=project_name
    )
    encoded_bundle = base64.b64encode(agent_code_bundle.agent_bundle).decode("utf-8")

    typer.echo(
        typer.style(
            "🚀 Preparing to upload the following agents:",
            fg=typer.colors.CYAN,
            bold=True,
        )
    )
    for name in agent_code_bundle.agent_names:
        typer.echo(
            typer.style(
                f"  • {project_name}:{name}", fg=typer.colors.MAGENTA, bold=True
            )
        )
    typer.echo(
        typer.style("These agents will be uploaded to tenant ", fg=typer.colors.BLUE)
        + typer.style(f"{config['tenant']}", fg=typer.colors.GREEN)
        + typer.style(" at ", fg=typer.colors.BLUE)
        + typer.style(f"{config['base_url']}", fg=typer.colors.GREEN)
    )
    confirm = typer.confirm(
        typer.style("Do you want to continue?", fg=typer.colors.CYAN), default=True
    )
    if not confirm:
        typer.echo(typer.style("Upload cancelled.", fg=typer.colors.RED))
        raise typer.Exit()

    host = config["base_url"]
    if not host.rstrip("/").endswith("v0/service"):
        host = host.rstrip("/") + "/v0/service"
    api_config = Configuration(host=host)
    api_client = ApiClient(api_config)
    api_client.set_default_header("X-Auth", config["api_key"])
    agent_bundles_api = AgentBundlesApi(api_client)

    try:
        existing_project = agent_bundles_api.retrieve_agent_bundle(
            user_roles="",
            user_tenants="",
            requester_uuid="",
            project=agent_code_bundle.project,
            tenant=config["tenant"],
        )
    except NotFoundException:
        existing_project = None

    if existing_project is not None:
        typer.echo(
            typer.style("⛔️ The project '", fg=typer.colors.YELLOW)
            + typer.style(agent_code_bundle.project, fg=typer.colors.GREEN)
            + typer.style(
                "' already exists. It contains the following agents:",
                fg=typer.colors.YELLOW,
            )
        )
        for agent in existing_project.agent_names:
            created_date = existing_project.created_at.strftime("%b %d, %Y %H:%M")
            updated_date = existing_project.last_updated_at.strftime("%b %d, %Y %H:%M")
            typer.echo(
                typer.style(
                    f"  • {existing_project.project}:{agent} ",
                    fg=typer.colors.MAGENTA,
                    bold=True,
                )
                + typer.style(
                    f"Created: {created_date} - Updated: {updated_date}",
                    dim=True,
                    italic=True,
                ),
            )
        confirm = typer.confirm(
            typer.style(
                "Do you want to overwrite the existing project?",
                fg=typer.colors.BRIGHT_YELLOW,
            ),
            default=False,
        )
        if not confirm:
            typer.echo(typer.style("Upload cancelled.", fg=typer.colors.RED))
            raise typer.Exit()
        else:
            agent_bundles_api.update_agent_bundle(
                user_roles="",
                user_tenants="",
                requester_uuid="",
                project=agent_code_bundle.project,
                tenant=config["tenant"],
                agent_bundle_patch=AgentBundlePatch(
                    agent_names=agent_code_bundle.agent_names,
                    agent_bundle=io.StringIO(encoded_bundle),
                ),
            )
    else:
        agent_bundles_api.create_agent_bundle(
            user_roles="",
            user_tenants="",
            requester_uuid="",
            tenant=config["tenant"],
            agent_bundle_form=AgentBundleForm(
                project=agent_code_bundle.project,
                agent_names=agent_code_bundle.agent_names,
                agent_bundle=io.StringIO(encoded_bundle),
            ),
        )

    console.print()
    console.print(
        f"  [green]✅ Project '{agent_code_bundle.project}' uploaded successfully.[/]"
    )
    console.print()

    project_config = ProjectConfig(project_dir)
    build_dir = os.path.join(project_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    for agent_name in agent_code_bundle.agent_names:
        agent_setup = project_config.get_agent(agent_name)
        agent_configuration = (
            agent_setup.get("agent_configuration", {}) if agent_setup else {}
        )

        bot_config_data = __build_agent_setup(
            agent_code_bundle.project, agent_name, agent_configuration
        )
        bot_config = json.dumps(bot_config_data, indent=2)

        output_file = os.path.join(build_dir, f"{agent_name}.json")
        with open(output_file, "w") as f:
            f.write(bot_config)
            f.write("\n")

        console.print(
            Panel(
                f"[bold]Agent Setup:[/]\n\n"
                f"[cyan]{bot_config}[/]\n\n"
                f"[dim]Saved to: {output_file}\n"
                f"Go to your tenant → Create Agent → use the above "
                f"to fill in the agent form fields.[/]",
                title=f"[bold]{agent_code_bundle.project}:{agent_name}[/]",
                border_style="green",
            )
        )

    console.print(f"  [dim]Agent setup files saved to: {build_dir}/[/]")
    console.print()
