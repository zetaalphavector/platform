import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from typing_extensions import Annotated

from zav.agents_sdk.cli.commands.agent_cmd import prompt_agent_setup
from zav.agents_sdk.cli.project_config import CONFIG_FILE, ProjectConfig
from zav.agents_sdk.cli.utils import scaffold_project
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

    scaffold_project(project_dir, __version__)
    config = ProjectConfig(project_dir)
    public_setup, secret_setup = prompt_agent_setup("agent", config)
    config.add_agent(public_setup, secret_setup)

    tree = Tree(f"[bold green]{project_dir}/[/]")
    tree.add(
        "[cyan]agent_setups.json[/]       ← agents (reference LLM configs by name)"
    )
    tree.add("[cyan]llm_configurations.json[/] ← named LLM configurations")
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
