import os
from typing import Optional, Tuple

import typer
from rich.console import Console
from typing_extensions import Annotated

from zav.agents_sdk.cli.project_config import (
    CONFIG_FILE,
    INSTRUCTIONS_DIR,
    SKILLS_DIR,
    SPECS_DIR,
    ProjectConfig,
)
from zav.agents_sdk.cli.require import resolve_project_dir
from zav.agents_sdk.cli.source_aliases import (
    infer_vendor,
)

console = Console()


def run_validation(project_dir: str, agent: Optional[str] = None) -> Tuple[int, int]:
    """Run validation checks and return (errors, warnings) count."""
    errors = 0
    warnings = 0

    console.print()
    console.print("[bold]Checking project configuration...[/]")
    console.print()

    config_path = os.path.join(project_dir, CONFIG_FILE)
    if not os.path.isfile(config_path):
        console.print(f"  [red]❌ {CONFIG_FILE} not found[/]")
        return (1, 0)
    console.print(f"  [green]✅ {CONFIG_FILE} exists[/]")

    try:
        config = ProjectConfig(project_dir)
    except Exception as e:
        console.print(f"  [red]❌ Failed to parse {CONFIG_FILE}: {e}[/]")
        return (1, 0)

    agents = config.list_agents()
    if not agents:
        console.print("  [red]❌ No agents defined in configuration[/]")
        errors += 1
    else:
        console.print(f"  [green]✅ {len(agents)} agent(s) defined[/]")

    try:
        agent_setup = config.get_agent_or_default(agent)
    except Exception:
        console.print("  [red]❌ Could not load agent configuration[/]")
        return (1, 0)

    vendor, model_name, model_cfg = config.get_model_config(
        agent_setup.get("agent_identifier")
    )
    if model_name:
        inferred = infer_vendor(model_name)
        if inferred and vendor and inferred != vendor:
            console.print(
                f"  [yellow]⚠️  Model '{model_name}' looks like '{inferred}' "
                f"but vendor is '{vendor}'[/]"
            )
            warnings += 1
        else:
            console.print(f"  [green]✅ Model {model_name} [dim]({vendor})[/][/]")
    else:
        console.print("  [red]❌ No model configured[/]")
        errors += 1

    identifier = agent_setup.get("agent_identifier", "")
    secret = config.get_secret(identifier)
    if secret:
        vendor_cfg = secret.get("llm_client_configuration", {}).get(
            "vendor_configuration", {}
        )
        has_key = False
        for v_cfg in vendor_cfg.values():
            if isinstance(v_cfg, dict):
                for k, v in v_cfg.items():
                    if "key" in k.lower() and v:
                        has_key = True
                        break
        if has_key:
            console.print("  [green]✅ API key configured in env/[/]")
        else:
            console.print("  [yellow]⚠️  No API key found in env/agent_setups.json[/]")
            warnings += 1
    else:
        console.print("  [yellow]⚠️  No secret configuration found for agent[/]")
        warnings += 1

    skills_dir = os.path.join(project_dir, SKILLS_DIR)
    if os.path.isdir(skills_dir):
        skills = config.list_skills()
        console.print(
            f"  [green]✅ {SKILLS_DIR}/ directory exists ({len(skills)} files)[/]"
        )
    else:
        console.print(f"  [dim]ℹ️  No {SKILLS_DIR}/ directory[/]")

    instr_dir = os.path.join(project_dir, INSTRUCTIONS_DIR)
    if os.path.isdir(instr_dir):
        instructions = config.list_instructions()
        console.print(
            f"  [green]✅ {INSTRUCTIONS_DIR}/ directory exists "
            f"({len(instructions)} files)[/]"
        )
    else:
        console.print(f"  [dim]ℹ️  No {INSTRUCTIONS_DIR}/ directory[/]")

    specs_dir = os.path.join(project_dir, SPECS_DIR)
    if os.path.isdir(specs_dir):
        specs = config.list_specs()
        console.print(
            f"  [green]✅ {SPECS_DIR}/ directory exists ({len(specs)} files)[/]"
        )
    else:
        console.print(f"  [dim]ℹ️  No {SPECS_DIR}/ directory[/]")

    agent_cfg = agent_setup.get("agent_configuration", {})
    memory_cfg = agent_cfg.get("memory_provider_configuration", {})
    if memory_cfg.get("enabled"):
        file_mem = agent_cfg.get("file_memory_store_configuration", {})
        if file_mem.get("enabled"):
            console.print(
                '  [yellow]⚠️  Memory store is "file" - '
                "use tag-notes for production[/]"
            )
            warnings += 1

    mcp_cfg = agent_cfg.get("mcp_tools_provider_configuration", {})
    mcp_servers = mcp_cfg.get("servers", [])
    for server in mcp_servers:
        url = server.get("url", "")
        if "localhost" in url or "127.0.0.1" in url:
            name = server.get("name", "unnamed")
            console.print(
                f'  [red]❌ MCP server "{name}" uses localhost URL '
                f"- will not work in production[/]"
            )
            errors += 1

    console.print()
    if errors == 0 and warnings == 0:
        console.print("  [bold green]All checks passed ✅[/]")
    else:
        parts = []
        if warnings:
            parts.append(f"[yellow]{warnings} warning(s)[/]")
        if errors:
            parts.append(f"[red]{errors} error(s)[/]")
        console.print(f"  {', '.join(parts)}")
    console.print()

    return (errors, warnings)


def validate_command(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(help="The project directory."),
    ] = None,
    agent: Annotated[
        Optional[str],
        typer.Option("--agent", help="Agent identifier (defaults to first agent)."),
    ] = None,
):
    """
    Validate project configuration. Catches misconfigs before dev/serve/upload.
    """
    project_dir = resolve_project_dir(project_dir)
    errors, warnings = run_validation(project_dir, agent)
    if errors > 0:
        raise typer.Exit(code=1)
