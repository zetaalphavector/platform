import asyncio
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from zav.agents_sdk.cli.project_config import INSTRUCTIONS_DIR
from zav.agents_sdk.cli.require import require_project, resolve_project_dir

console = Console()
instruction_app = typer.Typer(no_args_is_help=True)


@instruction_app.command("add")
def instruction_add(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Instruction name (used as filename)."),
    ] = None,
    description: Annotated[
        str,
        typer.Option(
            "--description",
            "-d",
            help="What the instruction should cover. Used as prompt for the agent.",
        ),
    ] = "",
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    no_generate: Annotated[
        bool,
        typer.Option(
            "--no-generate",
            help="Skip agent generation; create empty instruction file.",
        ),
    ] = False,
):
    """
    Create a new instruction file in instructions/.

    By default, the agent generates the instruction content. Use
    --no-generate to create an empty file.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("Instruction name")

    slug = name.lower().replace(" ", "-").replace("_", "-")
    filename = f"{slug}.md"
    instr_dir = os.path.join(project_dir, INSTRUCTIONS_DIR)
    os.makedirs(instr_dir, exist_ok=True)

    filepath = os.path.join(instr_dir, filename)
    if os.path.exists(filepath):
        console.print(f"  [yellow]Instruction '{filename}' already exists[/]")
        raise typer.Exit(code=1)

    if not description:
        description = typer.prompt(
            "Describe what this instruction should cover",
            default=f"Instructions for {name}",
        )

    if no_generate:
        __create_empty_instruction(filepath, name, filename)
        return

    project_dir_abs = os.path.abspath(project_dir)
    if project_dir_abs not in sys.path:
        sys.path.insert(0, project_dir_abs)

    import zav.agents_sdk.agents  # noqa: F401

    if os.path.isfile(os.path.join(project_dir_abs, "__init__.py")):
        from zav.agents_sdk.cli.load_chat_agent_factory import (
            from_string as import_chat_agent_class_registry_from_string,
        )

        import_chat_agent_class_registry_from_string(project_dir_abs)

    prompt = (
        f"Generate an instruction document for the following purpose: {description}. "
        f"Write the content as a markdown file that will be loaded into the agent's "
        f"system prompt. Be specific and actionable. Output ONLY the markdown content, "
        f"nothing else."
    )
    console.print("  [dim]Generating instruction via agent...[/]")

    try:
        from zav.agents_sdk.cli.commands.run_cmd import run_agent_headless

        response = asyncio.run(run_agent_headless(prompt, project_dir_abs))
        if response:
            with open(filepath, "w") as f:
                f.write(response.strip() + "\n")
            console.print(
                f"  [green]✅ Created {INSTRUCTIONS_DIR}/{filename} (agent-generated)[/]"
            )
        else:
            console.print(
                "  [yellow]Agent returned empty response. Creating empty file.[/]"
            )
            __create_empty_instruction(filepath, name, filename)
    except Exception as e:
        console.print(
            f"  [yellow]Agent generation failed ({e}). Creating empty file.[/]"
        )
        __create_empty_instruction(filepath, name, filename)


def __create_empty_instruction(filepath: str, name: str, filename: str):
    title = name.replace("-", " ").replace("_", " ").title()
    with open(filepath, "w") as f:
        f.write(f"# {title}\n\n")
    console.print(f"  [green]✅ Created {INSTRUCTIONS_DIR}/{filename}[/]")
    console.print("  [dim]Edit the file to add your instructions.[/]")


@instruction_app.command("list")
def instruction_list(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    List all instruction files.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    instructions = config.list_instructions()
    if not instructions:
        console.print("  [dim]No instruction files found in instructions/[/]")
        return
    console.print()
    for instr in instructions:
        console.print(f"  [cyan]{INSTRUCTIONS_DIR}/{instr}[/]")
    console.print(f"\n  [dim]{len(instructions)} instruction(s)[/]")
    console.print()


@instruction_app.command("remove")
def instruction_remove(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Instruction filename or name."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Remove an instruction file.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("Instruction name to remove")

    filename = name if name.endswith(".md") else f"{name}.md"
    filepath = os.path.join(project_dir, INSTRUCTIONS_DIR, filename)

    if not os.path.exists(filepath):
        console.print(f"  [red]Instruction '{filename}' not found[/]")
        raise typer.Exit(code=1)

    confirm = typer.confirm(f"Remove {INSTRUCTIONS_DIR}/{filename}?", default=False)
    if not confirm:
        console.print("  [dim]Cancelled[/]")
        raise typer.Exit()

    os.remove(filepath)
    console.print(f"  [green]✅ Removed {INSTRUCTIONS_DIR}/{filename}[/]")
