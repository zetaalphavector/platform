import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from zav.agents_sdk.cli.require import resolve_project_dir
from zav.agents_sdk.cli.utils import (
    create_dependency_files,
    init_dependencies,
    to_snake_case,
)

console = Console()
dependency_app = typer.Typer(no_args_is_help=True)


def __dep_exists_callback(dep_name_snake: str) -> str:
    console.print(f"  [yellow]Dependency '{dep_name_snake}' already exists.[/]")
    dep_name = typer.prompt("Enter a new dependency name")
    if not dep_name:
        raise typer.Exit()
    return dep_name


@dependency_app.command("add")
def dependency_add(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Dependency name."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Create a new injectable dependency module.

    This is an advanced feature for injecting custom services (HTTP clients,
    database connections, etc.) into agent handlers at startup.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("Dependency name")

    dependencies_dir = init_dependencies(project_dir=project_dir)
    dep_name_snake = create_dependency_files(
        project_dir=project_dir,
        dependencies_dir=dependencies_dir,
        dependency_name=name,
        dependency_file_exists_callback=__dep_exists_callback,
    )
    console.print(
        f"  [green]✅ Dependency '{dep_name_snake}' created in "
        f"{project_dir}/dependencies/[/]"
    )


@dependency_app.command("list")
def dependency_list(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    List dependency modules.
    """
    project_dir = resolve_project_dir(project_dir)
    deps_dir = os.path.join(project_dir, "dependencies")
    if not os.path.isdir(deps_dir):
        console.print("  [dim]No dependencies directory[/]")
        return

    py_files = sorted(
        f[:-3]
        for f in os.listdir(deps_dir)
        if f.endswith(".py") and f != "__init__.py" and not f.startswith(".")
    )

    if not py_files:
        console.print("  [dim]No dependency modules found[/]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("File", style="dim")

    for name in py_files:
        table.add_row(name, f"dependencies/{name}.py")

    console.print()
    console.print(table)
    console.print()


@dependency_app.command("remove")
def dependency_remove(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Dependency name to remove."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Remove a dependency module.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("Dependency name to remove")

    dep_snake = to_snake_case(name)
    filepath = os.path.join(project_dir, "dependencies", f"{dep_snake}.py")
    if not os.path.exists(filepath):
        console.print(f"  [red]Dependency '{dep_snake}' not found[/]")
        raise typer.Exit(code=1)

    confirm = typer.confirm(f"Remove dependency '{dep_snake}'?", default=False)
    if not confirm:
        console.print("  [dim]Cancelled[/]")
        raise typer.Exit()

    os.remove(filepath)
    console.print(f"  [green]✅ Dependency '{dep_snake}' removed[/]")
