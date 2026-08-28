import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from zav.agents_sdk.cli.require import is_project, resolve_any_project_dir
from zav.agents_sdk.cli.utils import (
    create_source_files,
    init_dependencies,
    scaffold_project,
    to_snake_case,
)
from zav.agents_sdk.version import __version__

console = Console()
tools_app = typer.Typer(no_args_is_help=True)

_DEFAULT_PROJECT_DIR = "agents"


def __file_exists_callback(existing: str) -> str:
    return typer.prompt(f"  Source '{existing}' already exists. Enter a different name")


def __resolve_or_default_project(project_dir: Optional[str]) -> str:
    if project_dir:
        return os.path.abspath(project_dir)
    for candidate in (".", "agents", "agent"):
        if is_project(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(_DEFAULT_PROJECT_DIR)


@tools_app.command("add")
def tools_add(
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
    """
    Scaffold a custom tool source.

    Creates a project skeleton if needed — no agent required. The tool is available
    to both agents and MCP servers configured in the same project.
    """
    target = __resolve_or_default_project(project_dir)
    if not is_project(target):
        scaffold_project(target, __version__)
        console.print(f"  [dim]Created project skeleton in {target}/[/]")

    dependencies_dir = init_dependencies(target)
    source_name = create_source_files(
        project_dir=target,
        dependencies_dir=dependencies_dir,
        source_name=name,
        provider_name="tools_llm" if with_llm else "tools",
        file_exists_callback=__file_exists_callback,
    )
    snake = to_snake_case(source_name)
    console.print(
        f"  [green]✅ Tool source '{snake}' created[/] "
        f"[dim]→ dependencies/{snake}.py[/]"
    )


@tools_app.command("list")
def tools_list(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """List the custom tool sources in this project."""
    project_dir = resolve_any_project_dir(project_dir)
    dependencies_dir = os.path.join(project_dir, "dependencies")
    sources = (
        sorted(
            f[:-3]
            for f in os.listdir(dependencies_dir)
            if f.endswith(".py") and f != "__init__.py"
        )
        if os.path.isdir(dependencies_dir)
        else []
    )

    if not sources:
        console.print("  [dim]No custom tool sources[/]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Tool source", style="cyan")
    table.add_column("File", style="dim")
    for source in sources:
        table.add_row(source, f"dependencies/{source}.py")

    console.print()
    console.print(table)
    console.print()


@tools_app.command("remove")
def tools_remove(
    name: Annotated[str, typer.Argument(help="Tool source to remove.")],
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """Remove a custom tool source (deletes its file and import)."""
    project_dir = resolve_any_project_dir(project_dir)
    snake = to_snake_case(name)
    source_file = os.path.join(project_dir, "dependencies", f"{snake}.py")
    if not os.path.isfile(source_file):
        console.print(f"  [red]Tool source '{snake}' not found[/]")
        raise typer.Exit(code=1)

    os.remove(source_file)
    __drop_import(os.path.join(project_dir, "__init__.py"), snake)
    console.print(f"  [green]✅ Tool source '{snake}' removed[/]")


def __drop_import(init_file: str, snake: str):
    if not os.path.isfile(init_file):
        return
    marker = f"from .dependencies.{snake} import *"
    with open(init_file, "r") as f:
        lines = f.readlines()
    with open(init_file, "w") as f:
        f.writelines(line for line in lines if marker not in line)
