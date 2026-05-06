import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from zav.agents_sdk.cli.require import (
    require_agent,
    require_project,
    resolve_agent_identifier,
    resolve_project_dir,
)
from zav.agents_sdk.cli.utils import (
    create_agent_files,
    to_snake_case,
)

console = Console()
agent_app = typer.Typer(no_args_is_help=True)


def __agent_exists_callback(agent_name_snake: str) -> str:
    console.print(f"  [yellow]Agent '{agent_name_snake}' already exists.[/]")
    agent_name = typer.prompt("Enter a new agent name")
    if not agent_name:
        raise typer.Exit()
    return agent_name


def __openai_key_prompt_callback(default_obscured: str) -> str:
    return typer.prompt("Enter your OpenAI API key", default=default_obscured)


def prompt_agent_setup(
    identifier: str,
    *,
    default_vendor: str = "openai",
    default_model: Optional[str] = None,
    skip_capabilities: bool = False,
    existing_config: Optional[dict] = None,
    existing_secret: Optional[dict] = None,
    project_dir: Optional[str] = None,
) -> tuple[dict, dict]:
    vendor = (
        typer.prompt(
            "LLM provider (openai / anthropic / azure_openai / ollama)",
            default=default_vendor,
        )
        .strip()
        .lower()
    )

    if default_model is None:
        default_model = "gpt-5.4-mini"
        if vendor == "anthropic":
            default_model = "claude-sonnet-4-6"
        elif vendor == "azure_openai":
            default_model = "gpt-4.1"
        elif vendor == "ollama":
            default_model = "llama3"
    model = typer.prompt("Model name", default=default_model)

    existing_api_key = ""
    if existing_secret:
        vendor_cfg = existing_secret.get("llm_client_configuration", {}).get(
            "vendor_configuration", {}
        )
        for v in vendor_cfg.values():
            for k, val in v.items():
                if "key" in k and val:
                    existing_api_key = val
                    break

    api_key_hint = " [••••••••]" if existing_api_key else ""
    api_key = typer.prompt(
        f"API key (stored in env/, never committed){api_key_hint}",
        hide_input=True,
        show_default=False,
        default=existing_api_key,
    )

    try:
        local_tz = os.readlink("/etc/localtime").split("/zoneinfo/")[-1]
    except (OSError, ValueError, IndexError):
        local_tz = "UTC"

    def __cap_default(provider_key: str, default: bool) -> bool:
        if existing_config is None:
            return default
        provider_cfg = existing_config.get(provider_key, {})
        return provider_cfg.get("enabled", default)

    if skip_capabilities:
        enable_tools = True
        enable_skills = True
        enable_memory = True
        enable_context = True
        enable_instructions = True
        enable_delegation = True
        enable_processing = True
        enable_dispatch = False
        enable_mcp = True
    else:
        console.print()
        console.print("  [bold]Capabilities[/]  [dim](toggle what the agent can do)[/]")
        enable_tools = typer.confirm(
            "  Enable tools?",
            default=__cap_default("tools_provider_configuration", True),
        )
        enable_skills = typer.confirm(
            "  Enable skills?",
            default=__cap_default("skills_provider_configuration", True),
        )
        enable_memory = typer.confirm(
            "  Enable memory?",
            default=__cap_default("memory_provider_configuration", True),
        )
        enable_context = typer.confirm(
            "  Enable context?",
            default=__cap_default("context_provider_configuration", True),
        )
        enable_instructions = typer.confirm(
            "  Enable instructions?",
            default=__cap_default("instructions_provider_configuration", True),
        )
        enable_delegation = typer.confirm(
            "  Enable agent delegation?",
            default=__cap_default("agent_delegation_provider_configuration", True),
        )
        enable_processing = typer.confirm(
            "  Enable message processing?",
            default=__cap_default("message_processing_provider_configuration", True),
        )
        enable_dispatch = typer.confirm(
            "  Enable dispatch?",
            default=__cap_default("dispatch_provider_configuration", False),
        )
        enable_mcp = typer.confirm(
            "  Enable MCP servers?",
            default=__cap_default("mcp_tools_provider_configuration", True),
        )

    agent_has_tenant = (
        True
        if existing_secret is None
        else bool(
            existing_secret
            and existing_secret.get("agent_configuration", {}).get("x_auth")
        )
    )

    existing_tenant_cfg: dict = {}
    if existing_config:
        base_url = existing_config.get("base_url", "")
        tenant = existing_config.get("tenant", "")
        if base_url or tenant:
            existing_tenant_cfg = {"base_url": base_url, "tenant": tenant}
    if existing_secret:
        x_auth = existing_secret.get("agent_configuration", {}).get("x_auth", "")
        if x_auth:
            existing_tenant_cfg["x_auth"] = x_auth

    tenant_config: dict = {}
    connect_tenant = typer.confirm(
        "Connect to a Zeta Alpha tenant? (enables search, documents, tags)",
        default=agent_has_tenant,
    )
    if connect_tenant:
        base_url = typer.prompt(
            "Platform API URL",
            default=existing_tenant_cfg.get("base_url") or "https://api.zeta-alpha.com",
        )
        existing_key = existing_tenant_cfg.get("x_auth", "")
        key_hint = " [••••••••]" if existing_key else ""
        tenant_api_key = typer.prompt(
            f"Platform API key{key_hint}",
            hide_input=True,
            show_default=False,
            default=existing_key,
        )
        tenant_name = typer.prompt(
            "Tenant name",
            default=existing_tenant_cfg.get("tenant", "zetaalpha"),
        )
        tenant_config = {
            "base_url": base_url,
            "x_auth": tenant_api_key,
            "tenant": tenant_name,
        }

    agent_configuration: dict = {
        "context_window_configuration": {"enabled": True, "preview_tokens": 3000},
        "tools_provider_configuration": {"enabled": enable_tools},
        "skills_provider_configuration": {
            "enabled": enable_skills,
            "injection_mode": "tools",
        },
        "memory_provider_configuration": {"enabled": enable_memory},
        "context_provider_configuration": {"enabled": enable_context},
        "instructions_provider_configuration": {"enabled": enable_instructions},
        "agent_delegation_provider_configuration": {"enabled": enable_delegation},
        "message_processing_provider_configuration": {"enabled": enable_processing},
        "dispatch_provider_configuration": {"enabled": enable_dispatch},
        "mcp_tools_provider_configuration": {"enabled": enable_mcp},
        "datetime_instruction_source_configuration": {"timezone": local_tz},
    }

    if enable_memory:
        agent_configuration["file_memory_store_configuration"] = {
            "enabled": True,
            "file_path": f"memories/{identifier}_memory.json",
        }
    if enable_skills:
        agent_configuration["disk_skills_source_configuration"] = {
            "enabled": True,
            "skills_directories": ["skills/"],
        }
        agent_configuration["skill_creation_skills_source_configuration"] = {
            "enabled": True,
        }

    public_setup = {
        "agent_identifier": identifier,
        "agent_name": "agent",
        "llm_client_configuration": {
            "vendor": vendor,
            "vendor_configuration": {},
            "model_configuration": {
                "name": model,
                "type": "chat",
                "temperature": 0.0,
            },
        },
        "agent_configuration": agent_configuration,
    }

    if tenant_config:
        agent_configuration["base_url"] = tenant_config["base_url"]
        agent_configuration["tenant"] = tenant_config["tenant"]
    else:
        agent_configuration["base_url"] = ""
        agent_configuration["tenant"] = ""

    vendor_config: dict = {}
    if vendor == "openai" and api_key:
        vendor_config = {"openai": {"openai_api_key": api_key, "openai_org": ""}}
    elif vendor == "anthropic" and api_key:
        vendor_config = {"anthropic": {"anthropic_api_key": api_key}}
    elif vendor == "azure_openai" and api_key:
        vendor_config = {
            "azure_openai": {
                "azure_openai_api_key": api_key,
                "azure_openai_endpoint": "",
                "azure_openai_api_version": "2024-02-15-preview",
            }
        }
    elif api_key:
        vendor_config = {vendor: {"api_key": api_key}}

    secret_setup: dict = {
        "agent_identifier": identifier,
        "llm_client_configuration": {"vendor_configuration": vendor_config},
    }

    if tenant_config:
        secret_setup["agent_configuration"] = {
            "x_auth": tenant_config["x_auth"],
        }
    else:
        secret_setup["agent_configuration"] = {"x_auth": ""}

    return public_setup, secret_setup


@agent_app.command("add")
def agent_add(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Agent identifier."),
    ] = None,
    custom: Annotated[
        bool,
        typer.Option("--custom", help="Create a custom Python agent class."),
    ] = False,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Create a new agent.

    By default, creates a config-based agent entry using the built-in agent
    class. No code needed. Use --custom to scaffold a Python ChatAgent
    subclass instead.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)

    if custom:
        if name is None:
            name = typer.prompt("Agent name", default="chat-agent")
        agent_name_snake = create_agent_files(
            project_dir=project_dir,
            agent_name=name,
            agent_exists_callback=__agent_exists_callback,
            openai_key_prompt_callback=__openai_key_prompt_callback,
        )
        console.print(
            f"  [green]✅ Custom agent '{agent_name_snake}' created in {project_dir}/[/]"
        )
        return

    if name is None:
        name = typer.prompt("Agent identifier", default="assistant")

    if config.get_agent(name) is not None:
        console.print(f"  [red]Agent '{name}' already exists[/]")
        raise typer.Exit(code=1)

    public_setup, secret_setup = prompt_agent_setup(name)
    config.add_agent(public_setup, secret_setup)

    vendor = public_setup["llm_client_configuration"]["vendor"]
    model = public_setup["llm_client_configuration"]["model_configuration"]["name"]
    console.print(f"  [green]✅ Agent '{name}' created ({vendor} / {model})[/]")


@agent_app.command("list")
def agent_list(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    List all configured agents.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    agents = config.list_agents()

    if not agents:
        console.print("  [dim]No agents configured[/]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Identifier", style="cyan")
    table.add_column("Agent Name")
    table.add_column("Type")
    table.add_column("Model", style="dim")

    for agent in agents:
        identifier = agent.get("agent_identifier", "?")
        agent_name = agent.get("agent_name", "?")
        is_builtin = agent_name == "agent"
        agent_type = "[green]built-in[/]" if is_builtin else "[yellow]custom[/]"
        model = (
            agent.get("llm_client_configuration", {})
            .get("model_configuration", {})
            .get("name", "?")
        )
        table.add_row(identifier, agent_name, agent_type, model)

    console.print()
    console.print(table)
    console.print()


@agent_app.command("show")
def agent_show(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Agent identifier."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Show an agent's configuration.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)

    if name is None:
        agent_setup = require_agent(config)
    else:
        found = config.get_agent(name)
        if found is None:
            console.print(f"  [red]Agent '{name}' not found[/]")
            raise typer.Exit(code=1)
        agent_setup = found

    identifier = agent_setup.get("agent_identifier", "?")
    agent_name = agent_setup.get("agent_name", "?")
    llm = agent_setup.get("llm_client_configuration", {})

    console.print()
    console.print(f"  [bold]Identifier:[/]  [cyan]{identifier}[/]")
    console.print(f"  [bold]Agent Name:[/]  {agent_name}")
    console.print(
        f"  [bold]Model:[/]       "
        f"{llm.get('model_configuration', {}).get('name', '?')}"
    )
    console.print(f"  [bold]Vendor:[/]      {llm.get('vendor', '?')}")
    console.print()


@agent_app.command("remove")
def agent_remove(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Agent identifier to remove."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Remove a custom agent.
    """
    project_dir = resolve_project_dir(project_dir)
    if name is None:
        name = typer.prompt("Agent identifier to remove")

    config = require_project(project_dir)
    agent_setup = config.get_agent(name)
    if agent_setup is None:
        console.print(f"  [red]Agent '{name}' not found[/]")
        raise typer.Exit(code=1)

    confirm = typer.confirm(f"Remove agent '{name}'?", default=False)
    if not confirm:
        console.print("  [dim]Cancelled[/]")
        raise typer.Exit()

    config.remove_agent(name)

    py_file = os.path.join(project_dir, f"{to_snake_case(name)}.py")
    if os.path.exists(py_file):
        os.remove(py_file)
        console.print(f"  [dim]Removed {py_file}[/]")

    console.print(f"  [green]✅ Agent '{name}' removed[/]")


@agent_app.command("configure")
def agent_configure(
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
    Reconfigure an existing agent (model, credentials, capabilities).
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    identifier = resolve_agent_identifier(config, agent)

    agent_setup = config.get_agent_or_default(identifier)
    llm = agent_setup.get("llm_client_configuration", {})
    agent_cfg = agent_setup.get("agent_configuration", {})
    secret = config.get_secret(identifier) or {}

    current_vendor = llm.get("vendor", "openai")
    current_model = llm.get("model_configuration", {}).get("name")

    public_setup, secret_setup = prompt_agent_setup(
        identifier,
        default_vendor=current_vendor,
        default_model=current_model,
        skip_capabilities=False,
        existing_config=agent_cfg,
        existing_secret=secret,
        project_dir=project_dir,
    )

    public_setup.pop("agent_identifier", None)
    config.update_agent(identifier, public_setup)
    config.update_secret(identifier, secret_setup)

    vendor = (
        llm_cfg["vendor"]
        if (llm_cfg := public_setup.get("llm_client_configuration"))
        else current_vendor
    )
    model = llm_cfg["model_configuration"]["name"] if llm_cfg else current_model
    console.print(
        f"  [green]✅ Agent '{identifier}' reconfigured ({vendor} / {model})[/]"
    )
