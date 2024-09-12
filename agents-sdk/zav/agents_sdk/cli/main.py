import json
import os
import sys
from importlib.metadata import distribution
from typing import Optional

import typer
import uvicorn
from streamlit.web import cli as stcli
from typing_extensions import Annotated

from zav.agents_sdk.version import __version__

app = typer.Typer(no_args_is_help=True)


def is_valid_project_directory(directory: Optional[str] = None) -> bool:
    if directory is None:
        return False
    init_file = os.path.join(directory, "__init__.py")
    if not os.path.isfile(init_file):
        return False
    with open(init_file, "r") as f:
        content = f.read()
        return "Zeta Alpha Agents SDK" in content


def to_camel_case(name: str) -> str:
    return "".join(
        word.capitalize() for word in name.replace("-", " ").replace("_", " ").split()
    )


def to_snake_case(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").replace(" ", "_").lower()


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


@app.command()
def new(
    agent_name: Annotated[
        Optional[str],
        typer.Argument(help="The name of the agent to create."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option(help="The project directory where the agent will be created."),
    ] = None,
):
    """
    Creates a new agent in the project directory.
    """
    project_dir = get_project_directory(project_dir)
    if agent_name is None:
        agent_name = typer.prompt("Enter the agent name", default="chat-agent")
        if not agent_name:
            typer.echo("Agent name cannot be empty.")
            raise typer.Exit()

    class_name = to_camel_case(agent_name)
    agent_name_snake = to_snake_case(agent_name)
    agent_file = os.path.join(project_dir, f"{agent_name_snake}.py")
    while os.path.exists(agent_file):
        typer.echo(f"Agent '{agent_name_snake}' already exists.")
        agent_name = typer.prompt("Enter a new agent name")
        if not agent_name:
            typer.echo("Agent name cannot be empty.")
            raise typer.Exit()
        class_name = to_camel_case(agent_name)
        agent_name_snake = to_snake_case(agent_name)
        agent_file = os.path.join(project_dir, f"{agent_name_snake}.py")

    # Check for existing OpenAI API key in env/agent_setups.json
    env_agent_setups_file = os.path.join(project_dir, "env", "agent_setups.json")
    default_openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if os.path.exists(env_agent_setups_file):
        with open(env_agent_setups_file, "r") as f:
            env_setups = json.load(f)
            for setup in env_setups:
                if "openai_api_key" in setup.get("llm_client_configuration", {}).get(
                    "vendor_configuration", {}
                ).get("openai", {}):
                    default_openai_api_key = setup["llm_client_configuration"][
                        "vendor_configuration"
                    ]["openai"]["openai_api_key"]
                    break

    # Ask for OpenAI API key
    default_openai_api_key_obscured = default_openai_api_key[:8] + "..."
    openai_api_key = typer.prompt(
        "Enter your OpenAI API key", default=default_openai_api_key_obscured
    )
    if openai_api_key == default_openai_api_key_obscured:
        openai_api_key = default_openai_api_key

    # Append import statement to __init__.py
    init_file = os.path.join(project_dir, "__init__.py")
    with open(init_file, "a") as f:
        f.write(f"\nfrom .{agent_name_snake} import *\n")

    # Create the agent file
    with open(agent_file, "w") as f:
        f.write(
            f"""from typing import AsyncGenerator, List

from zav.agents_sdk import ChatAgentFactory, ChatMessage, StreamableChatAgent
from zav.agents_sdk.adapters import ZAVChatCompletionClient


@ChatAgentFactory.register()
class {class_name}(StreamableChatAgent):
    agent_name = "{agent_name_snake}"

    def __init__(self, client: ZAVChatCompletionClient):
        self.client = client

    async def execute_streaming(
        self, conversation: List[ChatMessage]
    ) -> AsyncGenerator[ChatMessage, None]:
        response = await self.client.complete(
            messages=conversation,
            max_tokens=2048,
            stream=True,
        )
        async for chat_client_response in response:
            if chat_client_response.error is not None:
                raise chat_client_response.error
            if chat_client_response.chat_completion is None:
                raise Exception("No response from chat completion client")

            yield ChatMessage.from_orm(chat_client_response.chat_completion)
"""
        )

    # Update agent_setups.json
    agent_setups_file = os.path.join(project_dir, "agent_setups.json")
    with open(agent_setups_file, "r+") as f:
        setups = json.load(f)
        setups.append(
            {
                "agent_identifier": agent_name_snake,
                "agent_name": agent_name_snake,
                "llm_client_configuration": {
                    "vendor": "openai",
                    "vendor_configuration": {},
                    "model_configuration": {
                        "name": "gpt-4o-mini",
                        "type": "chat",
                        "temperature": 0.0,
                    },
                },
            }
        )
        f.seek(0)
        json.dump(setups, f, indent=2)
        f.truncate()

    # Update env/agent_setups.json
    with open(env_agent_setups_file, "r+") as f:
        env_setups = json.load(f)
        env_setups.append(
            {
                "agent_identifier": agent_name_snake,
                "llm_client_configuration": {
                    "vendor_configuration": {
                        "openai": {"openai_api_key": openai_api_key, "openai_org": ""}
                    }
                },
            }
        )
        f.seek(0)
        json.dump(env_setups, f, indent=2)
        f.truncate()

    typer.echo(f"Agent '{agent_name_snake}' created successfully in {project_dir}")


@app.command()
def new_dependency(
    dependency_name: Annotated[
        Optional[str],
        typer.Argument(help="The name of the dependency to create."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option(
            help="The project directory where the dependency will be created."
        ),
    ] = None,
):
    """
    Creates a new dependency in the project directory.
    """
    project_dir = get_project_directory(project_dir)
    dependencies_dir = os.path.join(project_dir, "dependencies")
    os.makedirs(dependencies_dir, exist_ok=True)

    if dependency_name is None:
        dependency_name = typer.prompt(
            "Enter the dependency name", default="url-crawler"
        )
        if not dependency_name:
            typer.echo("Dependency name cannot be empty.")
            raise typer.Exit()

    class_name = to_camel_case(dependency_name)
    dependency_name_snake = to_snake_case(dependency_name)
    dependency_file = os.path.join(dependencies_dir, f"{dependency_name_snake}.py")
    while os.path.exists(dependency_file):
        typer.echo(f"Dependency '{dependency_name_snake}' already exists.")
        dependency_name = typer.prompt("Enter a new dependency name")
        if not dependency_name:
            typer.echo("Dependency name cannot be empty.")
            raise typer.Exit()
        class_name = to_camel_case(dependency_name)
        dependency_name_snake = to_snake_case(dependency_name)
        dependency_file = os.path.join(dependencies_dir, f"{dependency_name_snake}.py")

    # Ensure __init__.py exists in dependencies directory
    init_file = os.path.join(dependencies_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w"):
            pass

    # Append import statement to project __init__.py
    project_init_file = os.path.join(project_dir, "__init__.py")
    with open(project_init_file, "a") as f:
        f.write(f"\nfrom .dependencies.{dependency_name_snake} import *\n")

    # Create the dependency file
    with open(dependency_file, "w") as f:
        f.write(
            f"""from typing import Dict, Optional
import httpx
from zav.agents_sdk import AgentDependencyFactory, AgentDependencyRegistry


class {class_name}:
    def __init__(self, headers: Optional[Dict[str, str]] = None):
        self.headers = headers

    async def crawl(self, url: str) -> str:
        \"\"\"Crawl the given URL and return the HTML content.\"\"\"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers)
            return response.text


class {class_name}Factory(AgentDependencyFactory):
    @classmethod
    def create(cls, headers: Optional[Dict[str, str]] = None) -> {class_name}:
        return {class_name}(headers=headers)


AgentDependencyRegistry.register({class_name}Factory)
"""
        )

    typer.echo(
        f"Dependency '{dependency_name_snake}' created successfully in {project_dir}"
    )


@app.command()
def init(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(
            help="The directory where the project will be initialized. If not provided,"
            " a wizard will ask for the directory name."
        ),
    ] = None,
):
    """
    Initializes a new Zeta Alpha Agents SDK project.
    """

    if project_dir is None:
        project_dir = typer.prompt("Enter the directory name", default="agents")

    os.makedirs(project_dir, exist_ok=True)

    # Create .gitignore file
    with open(os.path.join(project_dir, ".gitignore"), "w") as f:
        f.write("env/\n")

    # Create __init__.py file
    with open(os.path.join(project_dir, "__init__.py"), "w") as f:
        f.write(
            f'"""\nGenerated using Zeta Alpha Agents SDK Version: {__version__}\n"""\n'
        )

    # Create agent_setups.json file
    with open(os.path.join(project_dir, "agent_setups.json"), "w") as f:
        json.dump([], f)

    # Create env/agent_setups.json file
    env_project_dir = os.path.join(project_dir, "env")
    os.makedirs(env_project_dir, exist_ok=True)
    with open(os.path.join(env_project_dir, "agent_setups.json"), "w") as f:
        json.dump([], f)

    typer.echo(f"Project initialized in {project_dir}")

    # Run the new command to create an agent
    new(project_dir=project_dir)


@app.command()
def serve(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(help="The project directory where the agents are located."),
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
    project_dir = get_project_directory(project_dir)
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
        typer.Argument(help="The project directory where the agents are located."),
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
    project_dir = get_project_directory(project_dir)
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


@app.command()
def version():
    """
    Prints the current version of the SDK.
    """
    typer.echo(f"Zeta Alpha Agents SDK Version: {__version__}")


@app.callback()
def callback():
    pass
