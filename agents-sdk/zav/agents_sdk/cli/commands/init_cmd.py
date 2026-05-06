import json
import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from typing_extensions import Annotated

from zav.agents_sdk.cli.commands.agent_cmd import prompt_agent_setup
from zav.agents_sdk.cli.project_config import (
    CONFIG_FILE,
    ENV_DIR,
    PROJECT_DIRS,
)
from zav.agents_sdk.version import __version__

console = Console()


def init_command(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(help="Directory to create the project in."),
    ] = None,
):
    """
    Initialize a new Zeta Alpha agents project.

    Creates a config-first project with the built-in agent. No Python code
    needed. The project is immediately usable with 'serve' or 'dev'.
    """
    if project_dir is None:
        project_dir = typer.prompt("Project directory", default="agents")

    if os.path.isfile(os.path.join(project_dir, CONFIG_FILE)):
        console.print(f"[yellow]Project already exists in {project_dir}/[/]")
        raise typer.Exit()

    public_setup, secret_setup = prompt_agent_setup("agent")

    os.makedirs(project_dir, exist_ok=True)
    for subdir in PROJECT_DIRS:
        os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(project_dir, ENV_DIR), exist_ok=True)

    with open(os.path.join(project_dir, ".gitignore"), "w") as f:
        f.write("env/\nmemories/\n")

    with open(os.path.join(project_dir, "__init__.py"), "w") as f:
        f.write(
            f'"""\nGenerated using Zeta Alpha Agents SDK Version: {__version__}\n"""\n'
            "from zav.agents_sdk.agents.agent import *  # noqa: F401,F403\n"
        )

    with open(os.path.join(project_dir, CONFIG_FILE), "w") as f:
        json.dump([public_setup], f, indent=2)
        f.write("\n")

    with open(os.path.join(project_dir, ENV_DIR, CONFIG_FILE), "w") as f:
        json.dump([secret_setup], f, indent=2)
        f.write("\n")

    tree = Tree(f"[bold green]{project_dir}/[/]")
    tree.add("[cyan]agent_setups.json[/]       ← agent configuration")
    tree.add("[cyan]__init__.py[/]             ← extend with custom agent code")
    tree.add("[cyan]skills/[/]                  ← skill folders")
    tree.add("[cyan]specs/[/]                   ← behavior tests")
    tree.add("[dim]memories/[/]                  ← local memory (gitignored)")
    tree.add("[dim]env/[/]                       ← secrets (gitignored)")

    console.print()
    console.print(
        Panel(tree, title="[bold]✅ Project created[/]", border_style="green")
    )
    console.print()
    console.print("  [bold]Next steps:[/]")
    console.print(f"    cd {project_dir}")
    console.print("    za agents dev .           [dim]# start the dev environment[/]")
    console.print(
        "    za agents skill add       [dim]# add a skill (agent-generated)[/]"
    )
    console.print("    za agents run 'prompt'    [dim]# run agent headlessly[/]")
    console.print()
