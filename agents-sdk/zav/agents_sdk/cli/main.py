import json
import os
import sys
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from typing_extensions import Annotated

from zav.agents_sdk.cli.commands.agent_cmd import (
    agent_add,
    agent_configure,
    agent_list,
    agent_remove,
)
from zav.agents_sdk.cli.commands.dependency_cmd import dependency_app
from zav.agents_sdk.cli.commands.deploy_cmd import deploy_app
from zav.agents_sdk.cli.commands.init_cmd import init_command as new_init_command
from zav.agents_sdk.cli.commands.instruction_cmd import instruction_app
from zav.agents_sdk.cli.commands.mcp_cmd import mcp_app
from zav.agents_sdk.cli.commands.model_cmd import model_app
from zav.agents_sdk.cli.commands.perf_cmd import perf_app
from zav.agents_sdk.cli.commands.policy_cmd import policies_app
from zav.agents_sdk.cli.commands.provider_cmd import provider_app
from zav.agents_sdk.cli.commands.provider_factory import make_provider_app
from zav.agents_sdk.cli.commands.run_cmd import run_command
from zav.agents_sdk.cli.commands.show_cmd import show_command
from zav.agents_sdk.cli.commands.skill_cmd import skill_app
from zav.agents_sdk.cli.commands.source_cmd import source_app
from zav.agents_sdk.cli.commands.spec_cmd import spec_app
from zav.agents_sdk.cli.source_aliases import get_providers
from zav.agents_sdk.cli.utils import (
    create_agent_files,
    create_dependency_files,
    init_dependencies,
)

ZA_BASE_URL = "https://api.zeta-alpha.com"
console = Console()
app = typer.Typer(no_args_is_help=True)

# --- Sub-apps (order doesn't matter, registered at the bottom) ---

capabilities_app = typer.Typer(no_args_is_help=True)
for _prov_name, _prov_info in get_providers().items():
    if _prov_info.source_base is None:
        continue
    _display = _prov_info.display_name
    _cli_name = _prov_info.cli_name
    capabilities_app.add_typer(
        make_provider_app(_prov_name),
        name=_cli_name,
        help=f"Manage {_display.lower()}.",
        no_args_is_help=True,
    )
capabilities_app.add_typer(
    mcp_app,
    name="mcp-servers",
    help="Manage MCP server integrations.",
    no_args_is_help=True,
)

config_app = typer.Typer()


def get_value_or_prompt(prompt: str, default: str, error_message: str, **prompt_kwargs):
    def wrapper(value: Optional[str] = None) -> str:
        if value is None:
            value = typer.prompt(prompt, default=default, **prompt_kwargs)
            if not value:
                typer.echo(error_message)
                raise typer.Exit()
        return value

    return wrapper


def get_project_directory(project_dir: Optional[str] = None) -> str:
    from zav.agents_sdk.cli.require import resolve_project_dir

    return resolve_project_dir(project_dir)


def agent_exists_callback(agent_name_snake: str) -> str:
    typer.echo(f"Agent '{agent_name_snake}' already exists.")
    agent_name = typer.prompt("Enter a new agent name")
    if not agent_name:
        typer.echo("Agent name cannot be empty.")
        raise typer.Exit()
    return agent_name


def openai_key_prompt_callback(default_openai_api_key_obscured: str) -> str:
    openai_api_key = typer.prompt(
        "Enter your OpenAI API key", default=default_openai_api_key_obscured
    )
    return openai_api_key


@app.command(hidden=True)
def new(
    agent_name: Annotated[
        Optional[str],
        typer.Argument(
            callback=get_value_or_prompt(
                prompt="Enter the agent name",
                default="chat-agent",
                error_message="Agent name cannot be empty.",
            ),
            help="The name of the agent to create.",
        ),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option(
            callback=get_project_directory,
            help="The project directory where the agent will be created.",
        ),
    ] = None,
):
    """
    Creates a new agent in the project directory.
    """
    assert project_dir is not None
    assert agent_name is not None

    # Create new agent files
    agent_name_snake = create_agent_files(
        project_dir=project_dir,
        agent_name=agent_name,
        agent_exists_callback=agent_exists_callback,
        openai_key_prompt_callback=openai_key_prompt_callback,
    )
    typer.echo(
        typer.style(
            f"Agent '{agent_name_snake}' created successfully in {project_dir}",
            fg=typer.colors.GREEN,
        )
    )


def dependency_file_exists_callback(dependency_name_snake: str) -> str:
    typer.echo(f"Dependency '{dependency_name_snake}' already exists.")
    dependency_name = typer.prompt("Enter a new dependency name")
    if not dependency_name:
        typer.echo("Dependency name cannot be empty.")
        raise typer.Exit()
    return dependency_name


@app.command(hidden=True)
def new_dependency(
    dependency_name: Annotated[
        Optional[str],
        typer.Argument(
            callback=get_value_or_prompt(
                prompt="Enter the dependency name",
                default="url-crawler",
                error_message="Dependency name cannot be empty.",
            ),
            help="The name of the dependency to create.",
        ),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option(
            callback=get_project_directory,
            help="The project directory where the dependency will be created.",
        ),
    ] = None,
):
    """
    Creates a new dependency in the project directory.
    """
    assert project_dir is not None
    assert dependency_name is not None
    dependencies_dir = init_dependencies(project_dir=project_dir)

    dependency_name_snake = create_dependency_files(
        project_dir=project_dir,
        dependencies_dir=dependencies_dir,
        dependency_name=dependency_name,
        dependency_file_exists_callback=dependency_file_exists_callback,
    )

    typer.echo(
        typer.style(
            f"Dependency '{dependency_name_snake}' created successfully "
            f"in {project_dir}",
            fg=typer.colors.GREEN,
        )
    )


def serve(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(
            callback=get_project_directory,
            help="The project directory where the agents are located.",
        ),
    ] = None,
    setup_src: Annotated[
        Optional[str],
        typer.Option(help="Path of the agent setup configuration."),
    ] = None,
    secret_setup_src: Annotated[
        Optional[str],
        typer.Option(help="Path of the secret agent setup configuration."),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option(help="Enable auto-reload."),
    ] = False,
    host: Annotated[
        str,
        typer.Option(help="Host to listen on."),
    ] = "127.0.0.1",
    zav_fe_url: Annotated[
        str,
        typer.Option(help="Base URL of the Zeta Alpha Front End."),
    ] = "https://search.zeta-alpha.com",
):
    """
    Starts the local REST API server for the agents project.
    """
    assert project_dir is not None
    project_dir = os.path.abspath(project_dir)
    if setup_src is None:
        setup_src = os.path.join(project_dir, "agent_setups.json")
    if secret_setup_src is None:
        secret_setup_src = os.path.join(project_dir, "env", "agent_setups.json")

    os.environ["JSON_LOGGING"] = "0"
    os.environ["ZAV_FE_URL"] = zav_fe_url
    if project_dir:
        os.environ["ZAV_PROJECT_DIR"] = project_dir
    if setup_src:
        os.environ["ZAV_AGENT_SETUP_SRC"] = setup_src
    if secret_setup_src:
        os.environ["ZAV_SECRET_AGENT_SETUP_SRC"] = secret_setup_src
    # Keep the invocation directory importable so agents that use absolute
    # imports rooted at the app package (e.g. `from src.adapters import ...`)
    # still resolve after we chdir into the project directory below.
    invocation_dir = os.getcwd()
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [project_dir, invocation_dir, os.getenv("PYTHONPATH")])
    )
    sys.path.insert(0, invocation_dir)
    sys.path.insert(0, project_dir)
    os.chdir(project_dir)

    uvicorn.run(
        "zav.agents_sdk.cli.local_app:app",
        host=host,
        reload=reload,
    )


def dev(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(
            callback=get_project_directory,
            help="The project directory where the agents are located.",
        ),
    ] = None,
    setup_src: Annotated[
        Optional[str],
        typer.Option(help="Path of the agent setup configuration file."),
    ] = None,
    secret_setup_src: Annotated[
        Optional[str],
        typer.Option(help="Path of the secret agent setup configuration file."),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option(help="Enable auto-reload."),
    ] = False,
    host: Annotated[
        Optional[str],
        typer.Option(help="Host to listen on."),
    ] = None,
    zav_fe_url: Annotated[
        str,
        typer.Option(help="Base URL of the Zeta Alpha Front End."),
    ] = "https://search.zeta-alpha.com",
    storage_backend: Annotated[
        str,
        typer.Option(help="Storage backend for the generated files."),
    ] = "disk",
    storage_path: Annotated[
        str,
        typer.Option(help="Path to store the generated files."),
    ] = os.getcwd(),
):
    """
    Starts the Debugging Environment for the agents project.
    """
    assert project_dir is not None
    project_dir = os.path.abspath(project_dir)
    if setup_src is None:
        setup_src = os.path.join(project_dir, "agent_setups.json")
    if secret_setup_src is None:
        secret_setup_src = os.path.join(project_dir, "env", "agent_setups.json")

    os.environ["JSON_LOGGING"] = "0"
    os.environ["ZAV_FE_URL"] = zav_fe_url
    os.environ["STORAGE_BACKEND"] = storage_backend
    os.environ["STORAGE_PATH"] = storage_path
    if project_dir:
        os.environ["ZAV_PROJECT_DIR"] = project_dir
    if setup_src:
        os.environ["ZAV_AGENT_SETUP_SRC"] = setup_src
    if secret_setup_src:
        os.environ["ZAV_SECRET_AGENT_SETUP_SRC"] = secret_setup_src

    # Keep the invocation directory importable so agents that use absolute
    # imports rooted at the app package (e.g. `from src.adapters import ...`)
    # still resolve after we chdir into the project directory below.
    invocation_dir = os.getcwd()

    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [project_dir, invocation_dir, os.getenv("PYTHONPATH")])
    )

    sys.path.insert(0, invocation_dir)
    sys.path.insert(0, project_dir)
    os.chdir(project_dir)
    sys.argv = [
        "streamlit",
        "run",
        os.path.join(os.path.dirname(__file__), "ui_app.py"),
        *(
            [
                "--server.address",
                host,
            ]
            if host
            else []
        ),
        "--server.port",
        "8000",
        "--server.runOnSave",
        str(reload).lower(),
        "--server.fileWatcherType",
        "poll" if reload else "none",
        "--browser.gatherUsageStats",
        "false",
        "--client.showSidebarNavigation",
        "false",
    ]
    from streamlit.web import cli as stcli

    sys.exit(stcli.main())


def store_client_config(project_dir: str, base_url: str, api_key: str, tenant: str):
    config_path = os.path.join(project_dir, "env", "zav_config.json")
    config = {"base_url": base_url, "api_key": api_key, "tenant": tenant}

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as file:
        json.dump(config, file, indent=4)

    return config_path


def load_client_config(project_dir: str):
    config_path = os.path.join(project_dir, "env", "zav_config.json")

    with open(config_path, "r") as file:
        config = json.load(file)

    return config


@config_app.command("set")
def config_set(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(
            callback=get_project_directory,
            help="The project directory where the agents are located.",
        ),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            callback=get_value_or_prompt(
                prompt="Enter the Zeta Alpha API base URL",
                default=ZA_BASE_URL,
                error_message="Base URL cannot be empty.",
            ),
            help="The Zeta Alpha API base URL.",
        ),
    ] = None,
    tenant: Annotated[
        Optional[str],
        typer.Option(
            callback=get_value_or_prompt(
                prompt="Enter your Zeta Alpha tenant",
                default="",
                error_message="Tenant cannot be empty.",
            ),
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            callback=get_value_or_prompt(
                prompt="Enter your Zeta Alpha API key",
                default="",
                error_message="API key cannot be empty.",
                hide_input=True,
            ),
        ),
    ] = None,
):
    """
    Sets the client configuration for the project.
    """
    assert project_dir is not None
    assert base_url is not None
    assert tenant is not None
    assert api_key is not None

    config_path = store_client_config(
        project_dir=project_dir, base_url=base_url, api_key=api_key, tenant=tenant
    )
    typer.echo(
        typer.style(f"Configuration saved to {config_path}", fg=typer.colors.GREEN)
    )


@config_app.command("show")
def config_show(
    project_dir: Optional[str] = typer.Argument(
        None,
        callback=get_project_directory,
        help="The project directory where the configuration is stored.",
    ),
):
    """
    Shows the current configuration.

    The API key will only display its prefix for security.
    """
    assert project_dir is not None
    try:
        config = load_client_config(project_dir)
    except FileNotFoundError:
        typer.echo("No configuration found. Please run 'za platform login' first.")
        raise typer.Exit()

    base_url = config.get("base_url", "Not set")
    api_key = config.get("api_key", "")
    tenant = config.get("tenant", "")

    # Mask the API key: show only the first 8 characters
    if len(api_key) > 9:
        masked_api_key = api_key[:9] + "****"
    elif not api_key:
        masked_api_key = "Not set"
    else:
        masked_api_key = "****"

    typer.echo(typer.style("Base URL", fg=typer.colors.GREEN) + f": {base_url}")
    typer.echo(typer.style("Tenant", fg=typer.colors.GREEN) + f": {tenant}")
    typer.echo(typer.style("API Key", fg=typer.colors.GREEN) + f": {masked_api_key}")


@config_app.command("reset")
def config_reset(
    project_dir: Optional[str] = typer.Argument(
        None,
        callback=get_project_directory,
        help="The project directory where the configuration is stored.",
    ),
):
    """
    Resets the configuration to default values.

    The configuration file is updated with the default base URL and an empty API key.
    """
    assert project_dir is not None
    config_path = os.path.join(project_dir, "env", "zav_config.json")
    confirm = typer.confirm(
        "Are you sure you want to reset the configuration to default values?",
        default=False,
    )
    if not confirm:
        typer.echo(typer.style("Reset cancelled.", fg=typer.colors.RED))
        raise typer.Exit()

    default_config = {"base_url": ZA_BASE_URL, "api_key": "", "tenant": ""}

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as file:
        json.dump(default_config, file, indent=4)

    typer.echo(
        typer.style(
            f"Configuration reset to default at {config_path}", fg=typer.colors.GREEN
        )
    )


@app.callback()
def callback():
    pass


# --- Command registration (order determines help output) ---

app.command("init", rich_help_panel="Project")(new_init_command)
app.command("list", rich_help_panel="Agents")(agent_list)
app.command("show", rich_help_panel="Agents")(show_command)
app.command("add", rich_help_panel="Agents")(agent_add)
app.command("remove", rich_help_panel="Agents")(agent_remove)
app.command("configure", rich_help_panel="Agents")(agent_configure)
app.add_typer(
    model_app,
    name="model",
    help="Manage LLM model configuration.",
    rich_help_panel="Configuration",
)
app.add_typer(
    capabilities_app,
    name="capabilities",
    help="Manage agent capabilities (tools, skills, memory, etc.).",
    no_args_is_help=True,
    rich_help_panel="Configuration",
)
app.add_typer(
    policies_app,
    name="policies",
    help="Manage cross-cutting policies (citations, ...).",
    no_args_is_help=True,
    rich_help_panel="Configuration",
)
app.add_typer(
    skill_app,
    name="skill",
    help="Manage skill files.",
    no_args_is_help=True,
    rich_help_panel="Configuration",
)
app.command("dev", rich_help_panel="Development")(dev)
app.command("serve", rich_help_panel="Development")(serve)
app.command("run", rich_help_panel="Development")(run_command)
app.add_typer(spec_app, name="test", rich_help_panel="Development")
app.add_typer(
    perf_app,
    name="perf",
    help="Crash-test resumable streaming across servers (server + client).",
    no_args_is_help=True,
    rich_help_panel="Development",
)
app.add_typer(
    deploy_app,
    name="deploy",
    help="Bundle and deploy agents to the Zeta Alpha Platform.",
    no_args_is_help=True,
    rich_help_panel="Development",
)

# Hidden commands
app.add_typer(
    config_app,
    name="config",
    help="Zeta Alpha Client configuration.",
    no_args_is_help=True,
    hidden=True,
)
app.add_typer(
    provider_app,
    name="provider",
    help="Manage agent providers.",
    no_args_is_help=True,
    hidden=True,
)
app.add_typer(
    source_app,
    name="source",
    help="Manage sources within providers.",
    no_args_is_help=True,
    hidden=True,
)
app.add_typer(
    instruction_app,
    name="instruction",
    help="Manage instruction files.",
    no_args_is_help=True,
    hidden=True,
)
app.add_typer(
    dependency_app,
    name="dependency",
    help="Advanced: manage injectable dependencies.",
    no_args_is_help=True,
    hidden=True,
)
