import json
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

import typer
from click import ClickException
from pydantic_core import PydanticUndefined
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from zav.agents_sdk.cli.require import (
    require_project,
    resolve_agent_identifier,
    resolve_project_dir,
)

console = Console()


PolicyEntry = Tuple[str, str, Type]

_POLICIES_CACHE: Optional[List[PolicyEntry]] = None


def __get_policies() -> List[PolicyEntry]:
    global _POLICIES_CACHE
    if _POLICIES_CACHE is None:
        # Deferred import: adapters pull in heavy SDK internals (pydantic models,
        # LLM clients, etc.) — importing at module level adds ~2.5s to CLI startup.
        from zav.agents_sdk.adapters.agent_state.citation import CitationConfiguration

        _POLICIES_CACHE = [
            ("citations", "citation_configuration", CitationConfiguration),
        ]
    return _POLICIES_CACHE


def __get_policy(name: str) -> Optional[PolicyEntry]:
    for entry in __get_policies():
        if entry[0] == name:
            return entry
    return None


def __format_type(annotation) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return "Literal[" + ", ".join(repr(a) for a in args) + "]"
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return f"Optional[{__format_type(non_none[0])}]"
        return " | ".join(__format_type(a) for a in args)
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def __is_configured(agent_setup: Dict[str, Any], config_key: str) -> bool:
    return config_key in agent_setup and bool(agent_setup[config_key])


policies_app = typer.Typer(no_args_is_help=True)


@policies_app.callback()
def policies_callback():
    """
    Manage cross-cutting policies (citations, ...).
    """


@policies_app.command("list")
def policies_list(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    agent: Annotated[
        Optional[str],
        typer.Option("--agent", help="Agent identifier."),
    ] = None,
):
    """
    List all available policies and their status.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    identifier = resolve_agent_identifier(config, agent)
    agent_setup = config.get_agent_or_default(identifier)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Policy", style="cyan")
    table.add_column("Status")

    for name, config_key, model_cls in __get_policies():
        configured = __is_configured(agent_setup, config_key)
        status = "[green]active[/]" if configured else "[dim]inactive[/]"
        table.add_row(name, status)

    console.print()
    console.print(table)
    console.print()


@policies_app.command("show")
def policies_show(
    policy: Annotated[
        Optional[str],
        typer.Argument(help="Policy name (e.g. citations)."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    agent: Annotated[
        Optional[str],
        typer.Option("--agent", help="Agent identifier."),
    ] = None,
):
    """
    Show configuration for a policy.
    """
    policy = __resolve_policy_arg(policy)
    entry = __get_policy(policy)
    if entry is None:
        raise ClickException(f"Unknown policy: {policy}")
    _, config_key, model_cls = entry

    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    identifier = resolve_agent_identifier(config, agent)
    agent_setup = config.get_agent_or_default(identifier)

    cfg = agent_setup.get(config_key, {})
    configured = __is_configured(agent_setup, config_key)
    status = "[green]active[/]" if configured else "[dim]inactive[/]"

    console.print()
    console.print(f"  [bold]{policy}[/]  {status}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Value")
    table.add_column("Default", style="dim")
    table.add_column("Description", style="dim")

    for name, field_info in model_cls.model_fields.items():
        type_str = __format_type(field_info.annotation)
        current = cfg.get(name)
        default = field_info.default
        if default is PydanticUndefined:
            default = (
                field_info.default_factory() if field_info.default_factory else None
            )

        val_str = f"[green]{current}[/]" if current is not None else "[dim]-[/]"
        default_str = str(default) if default is not None else "-"
        desc = field_info.description or ""
        table.add_row(name, type_str, val_str, default_str, desc)

    console.print()
    console.print(table)
    console.print()


@policies_app.command("configure")
def policies_configure(
    policy: Annotated[
        Optional[str],
        typer.Argument(help="Policy name (e.g. citations)."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    agent: Annotated[
        Optional[str],
        typer.Option("--agent", help="Agent identifier."),
    ] = None,
):
    """
    Interactively configure a policy.
    """
    policy = __resolve_policy_arg(policy)
    entry = __get_policy(policy)
    if entry is None:
        raise ClickException(f"Unknown policy: {policy}")
    name, config_key, model_cls = entry

    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    identifier = resolve_agent_identifier(config, agent)
    agent_setup = config.get_agent_or_default(identifier)

    cfg = dict(agent_setup.get(config_key, {}))

    console.print()
    console.print(f"  [bold]Configure {name}[/]")
    console.print(
        "  [dim]Press Enter to keep current/default. Type 'null' to clear.[/]"
    )
    console.print()

    changed = False
    for field_name, field_info in model_cls.model_fields.items():
        type_str = __format_type(field_info.annotation)
        current = cfg.get(field_name)
        default = field_info.default
        if default is PydanticUndefined:
            default = (
                field_info.default_factory() if field_info.default_factory else None
            )

        display_current = current if current is not None else default
        hint = f" [{display_current}]" if display_current is not None else ""

        if field_info.description:
            console.print(f"  [dim]{field_info.description}[/]")

        raw = typer.prompt(
            f"  {field_name} ({type_str}){hint}",
            default="",
            show_default=False,
        )

        if raw == "":
            continue
        if raw.lower() == "null":
            if field_name in cfg:
                del cfg[field_name]
                changed = True
            continue

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = raw

        cfg[field_name] = parsed
        changed = True

    if changed:
        config.update_agent(identifier, {config_key: cfg})
        console.print()
        console.print(f"  [green]✅ {name} policy updated[/]")
    else:
        console.print()
        console.print("  [dim]No changes made[/]")


def __resolve_policy_arg(policy: Optional[str]) -> str:
    policies = __get_policies()
    available = [entry[0] for entry in policies]
    if policy is not None:
        if policy not in available:
            console.print(
                f"  [red]Unknown policy '{policy}'. "
                f"Available: {', '.join(available)}[/]"
            )
            raise typer.Exit(code=1)
        return policy

    console.print()
    console.print("  [bold]Available policies[/]")
    for i, (name, _, _) in enumerate(policies, start=1):
        console.print(f"  {i}. [cyan]{name}[/]")
    console.print()

    choice = typer.prompt("Select policy", default=policies[0][0]).strip()
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(policies):
            return policies[idx - 1][0]
        console.print(f"  [red]Invalid selection '{choice}'[/]")
        raise typer.Exit(code=1)

    if choice not in available:
        console.print(f"  [red]Unknown policy '{choice}'[/]")
        raise typer.Exit(code=1)
    return choice
