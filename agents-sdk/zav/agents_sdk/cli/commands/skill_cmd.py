import asyncio
import os
import shutil
import sys
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from zav.agents_sdk.cli.project_config import SKILLS_DIR
from zav.agents_sdk.cli.require import require_project, resolve_project_dir

console = Console()
skill_app = typer.Typer(no_args_is_help=True)


def __create_empty_skill(skill_dir: str, slug: str, description: str, name: str):
    title = name.replace("-", " ").replace("_", " ").title()
    content = (
        f"---\nname: {slug}\n"
        f"description: '{description}'\n---\n\n"
        f"# {title}\n\n"
        f"## Instructions\n\n"
        f"1. [First step]\n"
        f"2. [Second step]\n"
    )
    os.makedirs(skill_dir, exist_ok=True)
    with open(os.path.join(skill_dir, "Skill.md"), "w") as f:
        f.write(content)
    console.print(f"  [green]✅ Created {SKILLS_DIR}/{slug}/Skill.md[/]")


@skill_app.command("add")
def skill_add(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Skill name (used as directory name)."),
    ] = None,
    description: Annotated[
        str,
        typer.Option(
            "--description",
            "-d",
            help="What the skill should do. Used as prompt for the agent.",
        ),
    ] = "",
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    no_generate: Annotated[
        bool,
        typer.Option(
            "--no-generate", help="Skip agent generation; create empty skill file."
        ),
    ] = False,
):
    """
    Create a new skill.

    By default, the agent generates the skill content using its built-in
    create-skill capability. Use --no-generate to create an empty template.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("Skill name")

    slug = name.lower().replace(" ", "-").replace("_", "-")
    skill_dir = os.path.join(project_dir, SKILLS_DIR, slug)

    if os.path.exists(skill_dir):
        console.print(f"  [yellow]Skill '{slug}' already exists[/]")
        raise typer.Exit(code=1)

    if not description:
        description = typer.prompt(
            "Describe what this skill should do",
            default=f"Skill for {name}",
        )

    if no_generate:
        __create_empty_skill(skill_dir, slug, description, name)
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
        f'Create a skill named "{slug}" with the following purpose: {description}. '
        f"Use the create_skill tool to save it."
    )
    console.print("  [dim]Generating skill via agent...[/]")

    try:
        from zav.agents_sdk.cli.commands.run_cmd import run_agent_headless

        asyncio.run(run_agent_headless(prompt, project_dir_abs))
        if os.path.isdir(skill_dir):
            console.print(
                f"  [green]✅ Created {SKILLS_DIR}/{slug}/Skill.md (agent-generated)[/]"
            )
        else:
            console.print(
                "  [yellow]Agent did not create the skill file. "
                "Creating empty template.[/]"
            )
            __create_empty_skill(skill_dir, slug, description, name)
    except Exception as e:
        console.print(
            f"  [yellow]Agent generation failed ({e}). Creating empty template.[/]"
        )
        __create_empty_skill(skill_dir, slug, description, name)


@skill_app.command("list")
def skill_list(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    List all skill files.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    skills = config.list_skills()
    if not skills:
        console.print("  [dim]No skill files found in skills/[/]")
        return
    console.print()
    for skill in skills:
        console.print(f"  [cyan]{SKILLS_DIR}/{skill}[/]")
    console.print(f"\n  [dim]{len(skills)} skill(s)[/]")
    console.print()


@skill_app.command("remove")
def skill_remove(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Skill filename or name."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Remove a skill file.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("Skill name to remove")

    slug = name.replace(".md", "").lower().replace(" ", "-").replace("_", "-")
    skill_dir = os.path.join(project_dir, SKILLS_DIR, slug)

    if not os.path.isdir(skill_dir):
        console.print(f"  [red]Skill '{slug}' not found[/]")
        raise typer.Exit(code=1)

    confirm = typer.confirm(f"Remove {SKILLS_DIR}/{slug}/?", default=False)
    if not confirm:
        console.print("  [dim]Cancelled[/]")
        raise typer.Exit()

    shutil.rmtree(skill_dir)
    console.print(f"  [green]✅ Removed {SKILLS_DIR}/{slug}/[/]")
