import os
from typing import Any, Dict, Optional

import typer
from rich.console import Console

from zav.agents_sdk.cli.project_config import (
    CONFIG_FILE,
    ProjectConfig,
    ProjectConfigError,
)

console = Console()


def resolve_project_dir(project_dir: Optional[str]) -> str:
    if project_dir is not None:
        if not os.path.isfile(os.path.join(project_dir, CONFIG_FILE)):
            console.print(
                f"\n  [red]No agent project found in '{project_dir}'.[/]"
                "\n  Run [bold]za agents init[/] to create one.\n"
            )
            raise typer.Exit(code=1)
        return project_dir
    if os.path.isfile(CONFIG_FILE):
        return "."
    for candidate in ("agents", "agent"):
        if os.path.isfile(os.path.join(candidate, CONFIG_FILE)):
            return candidate
    project_dir = typer.prompt(
        "No agent project found. Enter the project directory",
        default="agents",
    )
    if not os.path.isfile(os.path.join(project_dir, CONFIG_FILE)):
        console.print(
            f"\n  [red]No agent project found in '{project_dir}'.[/]"
            "\n  Run [bold]za agents init[/] to create one.\n"
        )
        raise typer.Exit(code=1)
    return project_dir


def require_project(project_dir: str) -> ProjectConfig:
    config_path = os.path.join(project_dir, CONFIG_FILE)
    if not os.path.isfile(config_path):
        console.print(
            "\n  [red]No agent project found in this directory.[/]"
            "\n  Run [bold]za agents init[/] to create one.\n"
        )
        raise typer.Exit(code=1)
    return ProjectConfig(project_dir)


def require_agent(
    config: ProjectConfig, identifier: Optional[str] = None
) -> Dict[str, Any]:
    try:
        return config.get_agent_or_default(identifier)
    except ProjectConfigError:
        msg = "\n  [red]No agent found"
        if identifier:
            msg += f" with identifier '{identifier}'"
        msg += ".[/]\n  Run [bold]za agents init[/] to set up a project.\n"
        console.print(msg)
        raise typer.Exit(code=1)


def resolve_agent_identifier(config: ProjectConfig, agent: Optional[str] = None) -> str:
    agents = config.list_agents()
    if agent is not None:
        setup = config.get_agent(agent)
        if setup is None:
            console.print(f"\n  [red]Agent '{agent}' not found.[/]\n")
            raise typer.Exit(code=1)
        return agent
    if len(agents) == 1:
        return agents[0].get("agent_identifier", "agent")
    identifiers = [a.get("agent_identifier", "agent") for a in agents]
    console.print()
    console.print("  [bold]Available agents:[/]")
    for i, ident in enumerate(identifiers, 1):
        console.print(f"    {i}. {ident}")
    choice = typer.prompt("  Which agent?", default=identifiers[0])
    if choice in identifiers:
        return choice
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(identifiers):
            return identifiers[idx]
    except ValueError:
        pass
    console.print(f"  [red]Invalid choice '{choice}'[/]")
    raise typer.Exit(code=1)
