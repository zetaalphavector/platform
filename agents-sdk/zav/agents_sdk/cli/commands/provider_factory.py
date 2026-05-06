import json
from importlib import import_module
from typing import Literal, Optional, Union, get_args, get_origin

import typer
from pydantic_core import PydanticUndefined
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from zav.agents_sdk.cli.require import (
    require_agent,
    require_project,
    resolve_project_dir,
)
from zav.agents_sdk.cli.source_aliases import (
    SourceRegistry,
    get_provider_config_models,
    get_providers,
    get_source_config_key,
    get_source_config_model,
)
from zav.agents_sdk.cli.utils import (
    create_source_files,
    init_dependencies,
    to_snake_case,
)

console = Console()


def __load_provider_config_model(provider_name: str):
    entry = get_provider_config_models().get(provider_name)
    if entry is None:
        return None

    mod = import_module(entry[0])
    return getattr(mod, entry[1], None)


def __source_is_active(provider_name, internal, prov_cfg, agent_cfg):
    include = prov_cfg.get("include_sources", [])
    exclude = prov_cfg.get("exclude_sources", [])
    if include:
        return internal in include
    if exclude:
        return internal not in exclude
    source_config_key = get_source_config_key(provider_name, internal)
    source_cfg = agent_cfg.get(source_config_key, {})
    return source_cfg.get("enabled", False)


def __resolve_source(provider_name: str, source_arg: str) -> str:
    """Resolve a CLI source arg to internal name."""
    sources = SourceRegistry.get_sources(provider_name)
    normalized = source_arg.replace("-", "_")
    if normalized in sources:
        return normalized
    for meta in sources.values():
        if meta.cli_name == source_arg:
            return meta.source_name
    available = ", ".join(sorted(m.cli_name for m in sources.values()))
    console.print(f"  [red]Unknown source '{source_arg}'. Available: {available}[/]")
    raise typer.Exit(code=1)


def __select_source(
    provider_name: str, source_arg: Optional[str], action: str
) -> tuple[str, str]:
    sources = sorted(
        SourceRegistry.get_sources(provider_name).values(),
        key=lambda meta: meta.cli_name,
    )
    if not sources:
        console.print(f"  [red]No sources available for provider '{provider_name}'[/]")
        raise typer.Exit(code=1)

    if source_arg is not None:
        internal = __resolve_source(provider_name, source_arg)
        meta = SourceRegistry.get_source(provider_name, internal)
        cli_name = meta.cli_name if meta else source_arg
        return internal, cli_name

    if len(sources) == 1:
        meta = sources[0]
        console.print(
            f"  [dim]Using only available source '{meta.cli_name}' for {action}[/]"
        )
        return meta.source_name, meta.cli_name

    console.print()
    console.print(f"  [bold]Available sources to {action}[/]")
    for index, meta in enumerate(sources, start=1):
        console.print(f"  {index}. [cyan]{meta.cli_name}[/]")
    console.print()

    choice = typer.prompt(
        f"Select source to {action}",
        default=sources[0].cli_name,
    ).strip()
    if choice.isdigit():
        selected = int(choice)
        if 1 <= selected <= len(sources):
            meta = sources[selected - 1]
            return meta.source_name, meta.cli_name
        console.print(f"  [red]Invalid selection '{choice}'[/]")
        raise typer.Exit(code=1)

    internal = __resolve_source(provider_name, choice)
    meta = SourceRegistry.get_source(provider_name, internal)
    cli_name = meta.cli_name if meta else choice
    return internal, cli_name


def __show_provider_overview(provider_name, prov_key, display, project_dir, agent):
    config = require_project(project_dir)
    prov_cfg = config.get_provider_config(prov_key, agent) or {}

    console.print()
    console.print(f"  [bold]{display} Provider[/]")

    enabled = prov_cfg.get("enabled", False)
    status = "[green]enabled[/]" if enabled else "[dim]disabled[/]"
    console.print(f"  Status: {status}")

    for key, value in prov_cfg.items():
        if key in ("enabled", "include_sources", "exclude_sources"):
            continue
        console.print(f"  {key}: {value}")

    sources = SourceRegistry.get_sources(provider_name)
    if sources:
        agent_cfg = config.get_agent_config(agent)

        table = Table(show_header=True, header_style="bold")
        table.add_column("Source", style="cyan")
        table.add_column("Status")

        for meta in sorted(sources.values(), key=lambda m: m.cli_name):
            active = __source_is_active(
                provider_name, meta.source_name, prov_cfg, agent_cfg
            )
            src_status = "[green]active[/]" if active else "[dim]inactive[/]"
            table.add_row(meta.cli_name, src_status)

        console.print()
        console.print(table)
    console.print()


def __format_type(annotation) -> str:
    origin = get_origin(annotation)
    if origin is type(None):
        return "None"
    args = get_args(annotation)
    if origin is list or (hasattr(origin, "__name__") and origin.__name__ == "List"):
        inner = __format_type(args[0]) if args else "Any"
        return f"List[{inner}]"
    if origin is dict or (hasattr(origin, "__name__") and origin.__name__ == "Dict"):
        k = __format_type(args[0]) if args else "Any"
        v = __format_type(args[1]) if len(args) > 1 else "Any"
        return f"Dict[{k}, {v}]"
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return f"Optional[{__format_type(non_none[0])}]"
        return " | ".join(__format_type(a) for a in args)
    if origin is Literal:
        return "Literal[" + ", ".join(repr(a) for a in args) + "]"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def __prompt_config_fields(model_cls, current_cfg: dict) -> tuple[dict, bool]:
    """Interactively prompt for each field in a pydantic model.
    Returns (cfg, changed)."""
    cfg = dict(current_cfg)
    changed = False
    for name, field_info in model_cls.model_fields.items():
        if name in ("enabled", "include_sources", "exclude_sources", "servers"):
            continue

        current = cfg.get(name)
        default = field_info.default
        if default is PydanticUndefined:
            default = (
                field_info.default_factory() if field_info.default_factory else None
            )
        type_str = __format_type(field_info.annotation)

        desc = field_info.description
        if desc:
            console.print(f"  [dim]{desc}[/]")

        display_current = current if current is not None else default
        hint = f" [{display_current}]" if display_current is not None else ""
        raw = typer.prompt(
            f"  {name} ({type_str}){hint}",
            default="",
            show_default=False,
        )

        if raw == "":
            continue

        if raw.lower() == "null":
            if name in cfg:
                del cfg[name]
                changed = True
            continue

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = raw

        cfg[name] = parsed
        changed = True

    return cfg, changed


def __do_list_sources(provider_name, prov_key, display, project_dir, agent):
    config = require_project(project_dir)
    prov_cfg = config.get_provider_config(prov_key, agent) or {}
    agent_cfg = config.get_agent_config(agent)

    sources = SourceRegistry.get_sources(provider_name)
    if not sources:
        console.print(f"  [dim]No sources found for {display}[/]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Source", style="cyan")
    table.add_column("Internal Name", style="dim")
    table.add_column("Status")

    for meta in sorted(sources.values(), key=lambda m: m.cli_name):
        active = __source_is_active(
            provider_name, meta.source_name, prov_cfg, agent_cfg
        )
        src_status = "[green]active[/]" if active else "[dim]inactive[/]"
        table.add_row(meta.cli_name, meta.source_name, src_status)

    console.print()
    console.print(table)
    console.print()


def __do_show_source(provider_name, prov_key, project_dir, source, agent):
    internal, cli_name = __select_source(provider_name, source, "show")

    config = require_project(project_dir)
    agent_cfg = config.get_agent_config(agent)
    prov_cfg = config.get_provider_config(prov_key, agent) or {}

    source_config_key = get_source_config_key(provider_name, internal)
    source_cfg = agent_cfg.get(source_config_key, {})
    active = __source_is_active(provider_name, internal, prov_cfg, agent_cfg)

    console.print()
    status = "[green]active[/]" if active else "[dim]inactive[/]"
    console.print(f"  [bold]{cli_name}[/] [dim]({internal})[/]  {status}")

    model_cls = get_source_config_model(provider_name, internal)
    if model_cls is not None:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Field", style="cyan")
        table.add_column("Type", style="dim")
        table.add_column("Value")
        table.add_column("Default", style="dim")
        table.add_column("Description", style="dim")

        for name, field_info in model_cls.model_fields.items():
            if name == "enabled":
                continue
            type_str = __format_type(field_info.annotation)
            current = source_cfg.get(name)
            default = field_info.default

            if current is not None:
                val_str = f"[green]{current}[/]"
            else:
                val_str = "[dim]-[/]"

            default_str = str(default) if default is not None else "-"
            desc = field_info.description or ""
            table.add_row(name, type_str, val_str, default_str, desc)

        console.print()
        console.print(table)
    elif source_cfg:
        for key, value in source_cfg.items():
            console.print(f"    {key}: {value}")
    else:
        console.print("    [dim]No specific configuration[/]")
    console.print()


def __do_configure_source(provider_name, prov_key, project_dir, source, agent):
    internal, cli_name = __select_source(provider_name, source, "configure")

    config = require_project(project_dir)
    identifier = agent
    if identifier is None:
        identifier = require_agent(config).get("agent_identifier")

    model_cls = get_source_config_model(provider_name, internal)
    if model_cls is None:
        console.print(f"  [yellow]No configuration model found for '{cli_name}'[/]")
        return

    source_config_key = get_source_config_key(provider_name, internal)
    agent_cfg = config.get_agent_config(identifier)
    source_cfg = dict(agent_cfg.get(source_config_key, {}))

    console.print()
    console.print(f"  [bold]Configure {cli_name}[/]")
    console.print(
        "  [dim]Press Enter to keep current/default. Type 'null' to clear.[/]"
    )
    console.print()

    is_enabled = source_cfg.get("enabled", True)
    enable_choice = typer.confirm(f"  Enable {cli_name}?", default=is_enabled)
    changed = enable_choice != is_enabled
    if changed:
        source_cfg["enabled"] = enable_choice

    if not enable_choice:
        if changed:
            config.set_agent_config(source_config_key, source_cfg, identifier)
            console.print()
            console.print(f"  [green]✅ {cli_name} disabled[/]")
        else:
            console.print()
            console.print("  [dim]No changes made[/]")
        return

    source_cfg, fields_changed = __prompt_config_fields(model_cls, source_cfg)
    changed = changed or fields_changed

    if changed:
        config.set_agent_config(source_config_key, source_cfg, identifier)
        console.print()
        console.print(f"  [green]✅ {cli_name} configuration updated[/]")
    else:
        console.print()
        console.print("  [dim]No changes made[/]")


def __do_settings(provider_name, prov_key, display, project_dir, agent):
    config = require_project(project_dir)
    identifier = agent
    if identifier is None:
        identifier = require_agent(config).get("agent_identifier")

    model_cls = __load_provider_config_model(provider_name)
    if model_cls is None:
        console.print(f"  [yellow]No configuration model for {display} provider[/]")
        return

    prov_cfg = dict(config.get_provider_config(prov_key, identifier) or {})

    console.print()
    console.print(f"  [bold]{display} Provider Settings[/]")
    console.print(
        "  [dim]Press Enter to keep current/default. Type 'null' to clear.[/]"
    )
    console.print()

    is_enabled = prov_cfg.get("enabled", False)
    enable_choice = typer.confirm(f"  Enable {display} provider?", default=is_enabled)
    changed = enable_choice != is_enabled
    if changed:
        prov_cfg["enabled"] = enable_choice

    if not enable_choice:
        if changed:
            config.set_provider_config(prov_key, prov_cfg, identifier)
            console.print()
            console.print(f"  [green]✅ {display} provider disabled[/]")
        else:
            console.print()
            console.print("  [dim]No changes made[/]")
        return

    prov_cfg, fields_changed = __prompt_config_fields(model_cls, prov_cfg)
    changed = changed or fields_changed

    if changed:
        config.set_provider_config(prov_key, prov_cfg, identifier)
        console.print()
        console.print(f"  [green]✅ {display} provider settings updated[/]")
    else:
        console.print()
        console.print("  [dim]No changes made[/]")


def __do_add_source(provider_name, project_dir, name):
    require_project(project_dir)
    dependencies_dir = init_dependencies(project_dir)

    def file_exists_callback(existing: str) -> str:
        return typer.prompt(
            f"  Source '{existing}' already exists. Enter a different name"
        )

    source_name = create_source_files(
        project_dir=project_dir,
        dependencies_dir=dependencies_dir,
        source_name=name,
        provider_name=provider_name,
        file_exists_callback=file_exists_callback,
    )

    snake = to_snake_case(source_name)
    console.print()
    console.print(f"  [green]✅ Source '{snake}' created[/]")
    console.print(f"     [dim]→ dependencies/{snake}.py[/]")
    console.print()


def make_provider_app(provider_name: str) -> typer.Typer:
    prov_info = get_providers()[provider_name]
    prov_key = prov_info.config_key
    display = prov_info.display_name

    app = typer.Typer(invoke_without_command=True)

    @app.callback()
    def provider_default(
        ctx: typer.Context,
        project_dir: Annotated[
            Optional[str],
            typer.Option("--project-dir", help="Project directory."),
        ] = None,
        agent: Annotated[
            Optional[str],
            typer.Option("--agent", help="Agent identifier."),
        ] = None,
    ):
        if ctx.invoked_subcommand is not None:
            return
        project_dir = resolve_project_dir(project_dir)
        __show_provider_overview(provider_name, prov_key, display, project_dir, agent)

    @app.command("list")
    def list_sources(
        project_dir: Annotated[
            Optional[str],
            typer.Option("--project-dir", help="Project directory."),
        ] = None,
        agent: Annotated[
            Optional[str],
            typer.Option("--agent", help="Agent identifier."),
        ] = None,
    ):
        """List all available sources."""
        project_dir = resolve_project_dir(project_dir)
        __do_list_sources(provider_name, prov_key, display, project_dir, agent)

    @app.command("show")
    def show(
        source: Annotated[
            Optional[str],
            typer.Argument(help="Source name. If omitted, you will be prompted."),
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
        """Show configuration for a source."""
        project_dir = resolve_project_dir(project_dir)
        __do_show_source(provider_name, prov_key, project_dir, source, agent)

    @app.command("configure")
    def configure(
        source: Annotated[
            Optional[str],
            typer.Argument(help="Source name. If omitted, you will be prompted."),
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
        """Interactively configure a source."""
        project_dir = resolve_project_dir(project_dir)
        __do_configure_source(provider_name, prov_key, project_dir, source, agent)

    @app.command("settings")
    def settings(
        project_dir: Annotated[
            Optional[str],
            typer.Option("--project-dir", help="Project directory."),
        ] = None,
        agent: Annotated[
            Optional[str],
            typer.Option("--agent", help="Agent identifier."),
        ] = None,
    ):
        """Interactively configure provider-level settings."""
        project_dir = resolve_project_dir(project_dir)
        __do_settings(provider_name, prov_key, display, project_dir, agent)

    @app.command("add")
    def add_source(
        name: Annotated[
            str, typer.Argument(help="Name for the new source (e.g. my-custom-tools).")
        ],
        project_dir: Annotated[
            Optional[str],
            typer.Option("--project-dir", help="Project directory."),
        ] = None,
    ):
        """Scaffold a new custom source."""
        project_dir = resolve_project_dir(project_dir)
        __do_add_source(provider_name, project_dir, name)

    return app
