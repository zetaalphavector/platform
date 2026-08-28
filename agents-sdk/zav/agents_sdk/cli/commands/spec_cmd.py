import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from typing_extensions import Annotated

from zav.agents_sdk.cli.project_config import CONFIG_FILE, ENV_DIR, SPECS_DIR
from zav.agents_sdk.cli.require import require_project, resolve_project_dir

console = Console()
spec_app = typer.Typer(no_args_is_help=True)


def __resolve_test_paths(project_dir, specs_path, setup_src, secret_setup_src):
    if specs_path is None:
        specs_path = os.path.join(project_dir, SPECS_DIR)
    elif not os.path.isabs(specs_path):
        specs_path = os.path.join(project_dir, SPECS_DIR, specs_path)
    if setup_src is None:
        setup_src = os.path.join(project_dir, CONFIG_FILE)
    if secret_setup_src is None:
        secret_setup_src = os.path.join(project_dir, ENV_DIR, CONFIG_FILE)
    return specs_path, setup_src, secret_setup_src


def __setup_python_path(project_dir):
    existing_pythonpath = os.getenv("PYTHONPATH") or ""
    if project_dir not in existing_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join(
            filter(None, [project_dir, existing_pythonpath])
        )
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)


def __create_test_harness(specs_path, pattern, local_trace_store, harness_cls):
    harness = harness_cls(local_trace_store)
    specs_root = Path(specs_path)
    if specs_root.is_dir():
        harness.discover(specs_root, pattern=pattern)
    else:
        harness.add_spec(specs_root)
    harness.load()
    return harness


def __inject_tracing(agent_setup_retriever, local_trace_store):
    try:
        setups = asyncio.run(agent_setup_retriever.list())
        for s in setups:
            if s.tracing_configuration is None:
                agent_setup_retriever.update_agent_setup(
                    s.agent_identifier,
                    {
                        "tracing_configuration": {
                            "vendor": "capture",
                            "vendor_configuration": {
                                "capture": {"_store": local_trace_store}
                            },
                        }
                    },
                )
    except Exception:
        typer.echo(
            typer.style(
                "Failed to inject capture tracing config, continuing without it",
                fg=typer.colors.RED,
            )
        )


def __print_run_header(
    n_items,
    specs_path,
    pattern,
    project_dir,
    setup_src,
    secret_setup_src,
    debug,
    version,
):
    console.print(Rule("[bold cyan]Agent Spec Test Run[/]", style="cyan"))
    info_table = Table.grid(padding=(0, 1))
    info_table.add_row("[bold]collected[/]", f"[cyan]{n_items} items[/]")
    info_table.add_row("[bold]specs path[/]", f"[magenta]{specs_path}[/]")
    info_table.add_row("[bold]pattern[/]", f"[blue]{pattern}[/]")
    info_table.add_row("[bold]project dir[/]", f"[magenta]{project_dir}[/]")
    info_table.add_row("[bold]setup src[/]", f"[magenta]{setup_src}[/]")
    info_table.add_row("[bold]secret setup src[/]", f"[magenta]{secret_setup_src}[/]")
    info_table.add_row("[bold]SDK version[/]", f"[green]{version}[/]")
    traces_dir = None
    if debug:
        traces_dir = Path("agent-traces")
        traces_dir.mkdir(exist_ok=True)
        info_table.add_row("[bold]traces dir[/]", f"[yellow]{traces_dir.resolve()}[/]")
    console.print(Panel(info_table, border_style="cyan"))
    return traces_dir


def __print_failures(failures):
    console.print()
    console.print(Rule("[bold bright_red]FAILURES[/]", style="bright_red"))
    for ordinal, r in failures:
        header = f"[{ordinal:02d}] {r.spec_id}"
        console.print(f"[bold bright_red]{header}[/]")
        if r.description:
            console.print(f"   [dim italic]{r.description}[/]")
        if getattr(r, "path", None):
            console.print(f"   [dim]Spec: {r.path}[/]")
        if r.error:
            console.print(f"   [bright_yellow]Error: {r.error}[/]")
        if hasattr(r, "run_details") and r.run_details:
            console.print("   [dark_green]Run Details:[/]")
            formatted_json = json.dumps(r.run_details, indent=2)
            for line in formatted_json.split("\n"):
                console.print(f"   {line}")
        console.print()


@spec_app.callback()
def test_callback():
    """
    Manage and run behavior tests.
    """


@spec_app.command("run")
def test_run(
    specs_path: Annotated[
        Optional[str],
        typer.Argument(
            help="Path to a spec file or directory containing YAML spec files.",
        ),
    ] = None,
    pattern: Annotated[
        str,
        typer.Option(
            help="Glob pattern to match spec files inside the directory.",
            show_default=True,
        ),
    ] = "*.yaml",
    project_dir: Annotated[
        Optional[str],
        typer.Option(
            help="The project directory where the agents are located.",
        ),
    ] = None,
    setup_src: Annotated[
        Optional[str],
        typer.Option(help="Path of the agent setup configuration file."),
    ] = None,
    secret_setup_src: Annotated[
        Optional[str],
        typer.Option(help="Path of the secret agent setup configuration file."),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option(
            help="Dump per-spec trace files to agent-traces/ for debugging.",
        ),
    ] = False,
):
    """
    Run behavior-spec tests for the agents in the project.
    """
    from zav.llm_tracing import LocalTraceStore, TracingBackendFactory

    from zav.agents_sdk import AgentDependencyRegistry, AgentSetupRetrieverFromFile
    from zav.agents_sdk.behavior import TestHarness
    from zav.agents_sdk.cli.load_chat_agent_factory import (
        from_string as import_chat_agent_class_registry_from_string,
    )
    from zav.agents_sdk.version import __version__

    project_dir = resolve_project_dir(project_dir)
    specs_path, setup_src, secret_setup_src = __resolve_test_paths(
        project_dir, specs_path, setup_src, secret_setup_src
    )

    __setup_python_path(project_dir)
    import_chat_agent_class_registry_from_string(project_dir)

    local_trace_store = LocalTraceStore()
    harness = __create_test_harness(specs_path, pattern, local_trace_store, TestHarness)

    agent_setup_retriever = AgentSetupRetrieverFromFile(
        file_path=setup_src, secret_file_path=secret_setup_src
    )
    __inject_tracing(agent_setup_retriever, local_trace_store)

    # Deferred import: domain modules pull in heavy SDK internals — importing at
    # module level adds seconds to CLI startup for commands that don't need them.
    from zav.agents_sdk.domain.chat_agent_factory import ChatAgentFactory
    from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistry

    chat_agent_factory = ChatAgentFactory(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=ChatAgentClassRegistry,
        tracing_backend_factory=TracingBackendFactory,
        trace_state_params={},
        agent_dependency_registry=AgentDependencyRegistry,
        llm_configuration_store=agent_setup_retriever.llm_configuration_store,
    )

    n_items = len(list(harness.get_spec_paths()))
    traces_dir = __print_run_header(
        n_items,
        specs_path,
        pattern,
        project_dir,
        setup_src,
        secret_setup_src,
        debug,
        version=__version__,
    )

    failures = []
    results = []
    passed = 0
    total_time = 0.0
    idx = 0

    async def _inner():
        nonlocal idx, passed, total_time
        first = True
        async for case_result in harness.run_iter(chat_agent_factory):
            if case_result.status == "running":
                if not first:
                    console.print()
                first = False
                header = f"[{idx + 1:02d}] {case_result.spec_id}"
                console.print(f"[bold bright_blue]{header}[/]")
                if case_result.description:
                    console.print(f"   [dim italic]{case_result.description}[/]")
                if getattr(case_result, "path", None):
                    console.print(f"   [dim]Spec: {case_result.path}[/]")
            else:
                idx += 1
                results.append(case_result)
                total_time += case_result.duration or 0.0
                duration_str = (
                    f"[bright_magenta]{case_result.duration:.2f}s[/]"
                    if case_result.duration is not None
                    else ""
                )
                if debug:
                    trace_file = traces_dir / f"{case_result.spec_id}.trace.json"
                    trace_file.write_text(
                        json.dumps(
                            local_trace_store.export(),
                            indent=2,
                            default=str,
                        )
                    )
                    local_trace_store.reset()
                if case_result.status == "passed":
                    passed += 1
                    console.print(f"[bold green]PASSED[/] {duration_str}")
                    if debug:
                        console.print(f"   [dim]Trace: {trace_file}[/]")
                else:
                    failures.append((idx, case_result))
                    console.print(f"[bold bright_red]FAILED[/] {duration_str}")
                    if debug:
                        console.print(f"   [dim]Trace: {trace_file}[/]")
        console.print()

    asyncio.run(_inner())

    if failures:
        __print_failures(failures)

    failed = len(failures)
    total = len(results)
    summary_line = (
        f"[green]{passed} passed[/]"
        + (f", [bright_red]{failed} failed[/]" if failed else "")
        + f", [cyan]{total} total[/]"
    )
    console.print(
        Rule("[bold]SUMMARY[/]", style="green" if failed == 0 else "bright_red")
    )
    console.print(f"{summary_line} in [bold]{total_time:.2f}s[/]")
    if failed:
        raise typer.Exit(code=1)


SPEC_TEMPLATE = """id: {slug}
description: {description}
agent_identifier: agent
messages:
  - role: user
    content: "{prompt}"
expectations:
  - type: text_includes
    value: ""
"""


@spec_app.command("add")
def spec_add(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Spec name (used as filename)."),
    ] = None,
    description: Annotated[
        str,
        typer.Option("--description", "-d", help="Short description of the test."),
    ] = "",
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Create a new behavior test spec in specs/.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("Spec name")

    slug = name.lower().replace(" ", "-").replace("_", "-")
    filename = f"{slug}.yaml"
    specs_dir = os.path.join(project_dir, SPECS_DIR)
    os.makedirs(specs_dir, exist_ok=True)

    filepath = os.path.join(specs_dir, filename)
    if os.path.exists(filepath):
        console.print(f"  [yellow]Spec '{filename}' already exists[/]")
        raise typer.Exit(code=1)

    if not description:
        description = typer.prompt("Description", default=f"Test for {name}")

    prompt = typer.prompt("Example user message", default="Hello!")

    content = SPEC_TEMPLATE.format(slug=slug, description=description, prompt=prompt)
    with open(filepath, "w") as f:
        f.write(content)

    console.print(f"  [green]✅ Created {SPECS_DIR}/{filename}[/]")


@spec_app.command("list")
def spec_list(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    List all test spec files.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    specs = config.list_specs()
    if not specs:
        console.print("  [dim]No spec files found in specs/[/]")
        return
    console.print()
    for spec in specs:
        console.print(f"  [cyan]{SPECS_DIR}/{spec}[/]")
    console.print(f"\n  [dim]{len(specs)} spec(s)[/]")
    console.print()


@spec_app.command("remove")
def spec_remove(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Spec filename or name."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Remove a test spec file.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("Spec name to remove")

    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    filepath = os.path.join(project_dir, SPECS_DIR, filename)

    if not os.path.exists(filepath):
        console.print(f"  [red]Spec '{filename}' not found[/]")
        raise typer.Exit(code=1)

    confirm = typer.confirm(f"Remove {SPECS_DIR}/{filename}?", default=False)
    if not confirm:
        console.print("  [dim]Cancelled[/]")
        raise typer.Exit()

    os.remove(filepath)
    console.print(f"  [green]✅ Removed {SPECS_DIR}/{filename}[/]")
