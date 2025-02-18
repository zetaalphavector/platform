import base64
import io
import json
import os
import sys
from importlib.metadata import distribution
from typing import Optional

import typer
import uvicorn
from streamlit.web import cli as stcli
from typing_extensions import Annotated
from zav.chat_service import ApiClient, Configuration
from zav.chat_service.apis import AgentBundlesApi
from zav.chat_service.exceptions import NotFoundException
from zav.chat_service.models import AgentBundleForm, AgentBundlePatch

from zav.agents_sdk.cli.utils import (
    create_agent_files,
    create_dependency_files,
    init_dependencies,
    init_project,
    is_valid_project_directory,
)
from zav.agents_sdk.domain.agent_code_bundle import AgentCodeBundle
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistry
from zav.agents_sdk.version import __version__

ZA_BASE_URL = "https://api.zeta-alpha.com"
app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer()
app.add_typer(
    config_app,
    name="config",
    help="Zeta Alpha Client configuration.",
    no_args_is_help=True,
)


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
    if project_dir is None:
        project_dir = os.getcwd()
        if not is_valid_project_directory(project_dir):
            project_dir = typer.prompt("Enter the project directory", default="agents")
    if not is_valid_project_directory(project_dir):
        typer.echo(
            "Invalid project directory. Please ensure you are in a valid project "
            "directory."
        )
        typer.echo(
            typer.style(
                "You can create a new project directory by running 'rag_agents init'.",
                fg=typer.colors.GREEN,
            )
        )
        raise typer.Exit()
    return project_dir  # type: ignore


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


@app.command()
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


@app.command()
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

    # Create the dependency files
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


@app.command()
def init(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(
            callback=get_value_or_prompt(
                prompt="Enter the directory name",
                default="agents",
                error_message="Directory name cannot be empty.",
            ),
            help="The directory where the project will be initialized. If not provided,"
            " a wizard will ask for the directory name.",
        ),
    ] = None,
):
    """
    Initializes a new Zeta Alpha Agents SDK project.
    """
    assert project_dir is not None

    init_project(project_dir=project_dir, sdk_version=__version__)
    typer.echo(
        typer.style(f"Project initialized in {project_dir}", fg=typer.colors.GREEN)
    )

    # Run the new command to create an agent
    new(
        project_dir=project_dir,
        agent_name=get_value_or_prompt(
            prompt="Enter the agent name",
            default="chat-agent",
            error_message="Agent name cannot be empty.",
        )(),
    )


@app.command()
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
    # This is needed so the agent module can be reached inside the uvicorn process
    sys.path.insert(0, os.getcwd())

    uvicorn.run(
        "zav.agents_sdk.cli.local_app:app",
        host=host,
        reload=reload,
    )


@app.command()
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

    existing_pythonpath = os.getenv("PYTHONPATH")
    current_path = os.getcwd()
    os.environ["PYTHONPATH"] = (
        f"{existing_pythonpath}:{current_path}" if existing_pythonpath else current_path
    )

    # This is needed so the agent module can be reached inside the uvicorn process
    sys.path.insert(0, os.getcwd())
    sys.argv = [
        "streamlit",
        "run",
        str(distribution("zetaalpha.rag-agents").locate_file("zav/agents_sdk/cli/ui_app.py")),
        "--server.port",
        "8000",
        "--server.runOnSave",
        str(reload).lower(),
        "--server.fileWatcherType",
        "poll",
        "--browser.gatherUsageStats",
        "false",
        "--client.showSidebarNavigation",
        "false",
    ]
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


def create_agent_code_bundle(project_dir: str, project_name: str):
    # Update PYTHONPATH so agent modules can be discovered
    existing_pythonpath = os.getenv("PYTHONPATH")
    current_path = os.getcwd()
    os.environ["PYTHONPATH"] = (
        f"{existing_pythonpath}:{current_path}" if existing_pythonpath else current_path
    )

    # Load registries for the selected project
    AgentCodeBundle.load_agent_registries_from(project_dir=project_dir)
    agent_names = [
        agent.agent_name for agent in ChatAgentClassRegistry.registry.values()
    ]

    # Create a bundle with the provided project name
    agent_code_bundle = AgentCodeBundle.from_project_dir(
        project=project_name, agent_names=agent_names, project_dir=project_dir
    )
    return agent_code_bundle


@app.command()
def bundle(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(
            callback=get_project_directory,
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

    This can be uploaded to the Zeta Alpha Platform via the REST API. If you want to
    upload the project directly, use the 'upload' command instead.
    """
    assert project_dir is not None

    # Determine the project name to use for the upload
    if project_name is None:
        project_name = os.path.basename(os.path.abspath(project_dir))

    agent_code_bundle = create_agent_code_bundle(
        project_dir=project_dir, project_name=project_name
    )
    if output_dir is None:
        output_dir = project_dir

    # Store the agent bundle on disk
    with open(os.path.join(output_dir, "build.zip"), "wb") as file:
        file.write(agent_code_bundle.agent_bundle)


@app.command()
def upload(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(
            callback=get_project_directory,
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
    assert project_dir is not None

    # Determine the project name to use for the upload
    if project_name is None:
        project_name = os.path.basename(os.path.abspath(project_dir))

    # Get configuration
    try:
        config = load_client_config(project_dir)
    except FileNotFoundError:
        config_set(
            project_dir=project_dir,
            base_url=get_value_or_prompt(
                prompt="Enter the Zeta Alpha API base URL",
                default=ZA_BASE_URL,
                error_message="Base URL cannot be empty.",
            )(),
            api_key=get_value_or_prompt(
                prompt="Enter your Zeta Alpha API key",
                default="",
                error_message="API key cannot be empty.",
                hide_input=True,
            )(),
        )
        config = load_client_config(project_dir)

    agent_code_bundle = create_agent_code_bundle(
        project_dir=project_dir, project_name=project_name
    )
    encoded_bundle = base64.b64encode(agent_code_bundle.agent_bundle).decode("utf-8")

    # Prompt the user to verify if they want to upload the agents
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

    # Prepare API client with authentication configuration
    host = config["base_url"]
    if not host.rstrip("/").endswith("v0/service"):
        host = host.rstrip("/") + "/v0/service"
    api_config = Configuration(host=host)
    api_client = ApiClient(api_config)
    api_client.set_default_header("X-Auth", config["api_key"])
    agent_bundles_api = AgentBundlesApi(api_client)

    # First check if the project already exists
    try:
        existing_project = agent_bundles_api.retrieve_agent_bundle(
            # user_roles=json.dumps(
            #     dict(
            #         roles=["admin"],
            #         role_data=[],
            #     )
            # ),
            # user_tenants=json.dumps(dict(tenants=["test_tenant"])),
            # requester_uuid="test_requester_uuid",
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
                # user_roles=json.dumps(
                #     dict(
                #         roles=["admin"],
                #         role_data=[],
                #     )
                # ),
                # user_tenants=json.dumps(dict(tenants=["test_tenant"])),
                # requester_uuid="test_requester_uuid",
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
            # user_roles=json.dumps(
            #     dict(
            #         roles=["admin"],
            #         role_data=[],
            #     )
            # ),
            # user_tenants=json.dumps(dict(tenants=["test_tenant"])),
            # requester_uuid="test_requester_uuid",
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
    typer.echo(
        typer.style(
            f"Project '{agent_code_bundle.project}' was successfully uploaded "
            f"from '{project_dir}'.",
            fg=typer.colors.GREEN,
        )
    )


@app.command()
def list_remote(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(
            callback=get_project_directory,
            help="The project directory where the agents are located.",
        ),
    ] = None
):
    """
    Lists the projects and agents available on the Zeta Alpha Platform.
    """
    assert project_dir is not None

    # Get configuration
    try:
        config = load_client_config(project_dir)
    except FileNotFoundError:
        config_set(
            project_dir=project_dir,
            base_url=get_value_or_prompt(
                prompt="Enter the Zeta Alpha API base URL",
                default=ZA_BASE_URL,
                error_message="Base URL cannot be empty.",
            )(),
            api_key=get_value_or_prompt(
                prompt="Enter your Zeta Alpha API key",
                default="",
                error_message="API key cannot be empty.",
                hide_input=True,
            )(),
        )
        config = load_client_config(project_dir)

    # Prepare API client with authentication configuration
    host = config["base_url"]
    if not host.rstrip("/").endswith("v0/service"):
        host = host.rstrip("/") + "/v0/service"
    api_config = Configuration(host=host)
    api_client = ApiClient(api_config)
    api_client.set_default_header("X-Auth", config["api_key"])
    agent_bundles_api = AgentBundlesApi(api_client)

    # First check if the project already exists
    available_agent_bundles = agent_bundles_api.filter_agent_bundle(
        # user_roles=json.dumps(
        #     dict(
        #         roles=["admin"],
        #         role_data=[],
        #     )
        # ),
        # user_tenants=json.dumps(dict(tenants=["test_tenant"])),
        # requester_uuid="test_requester_uuid",
        user_roles="",
        user_tenants="",
        requester_uuid="",
        tenant=config["tenant"],
    )
    if available_agent_bundles.count == 0:
        typer.echo("No projects found.")
        return

    typer.echo(
        typer.style(
            "Agents available on the Zeta Alpha Platform for tenant ",
            fg=typer.colors.BLUE,
        )
        + typer.style(f"{config['tenant']}", fg=typer.colors.GREEN)
        + typer.style(" at ", fg=typer.colors.BLUE)
        + typer.style(f"{config['base_url']}", fg=typer.colors.GREEN)
        + typer.style(":", fg=typer.colors.BLUE)
    )
    for agent_bundle in available_agent_bundles.results:
        for agent in agent_bundle.agent_names:
            created_date = agent_bundle.created_at.strftime("%b %d, %Y %H:%M")
            updated_date = agent_bundle.last_updated_at.strftime("%b %d, %Y %H:%M")
            typer.echo(
                typer.style(
                    f"  • {agent_bundle.project}:{agent} ",
                    fg=typer.colors.MAGENTA,
                    bold=True,
                )
                + typer.style(
                    f"Created: {created_date} - Updated: {updated_date}",
                    dim=True,
                    italic=True,
                ),
            )


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
        typer.echo("No configuration found. Please run 'rag_agents config set' first.")
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


@app.command()
def version():
    """
    Prints the current version of the SDK.
    """
    typer.echo(f"Zeta Alpha Agents SDK Version: {__version__}")


@app.callback()
def callback():
    pass
