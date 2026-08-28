import json
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated
from zav.llm_domain import LLMModelConfiguration

from zav.agents_sdk.cli.project_config import ProjectConfig
from zav.agents_sdk.cli.require import (
    require_project,
    resolve_agent_identifier,
    resolve_any_project_dir,
    resolve_project_dir,
)
from zav.agents_sdk.cli.source_aliases import infer_vendor

console = Console()


VENDOR_REQUIRED_FIELDS: Dict[str, Dict[str, str]] = {
    "openai": {
        "openai_api_key": "OpenAI API key",
        "openai_org": "OpenAI organization ID",
    },
    "anthropic": {
        "anthropic_api_key": "Anthropic API key",
    },
    "azure_openai": {
        "endpoint": "Azure endpoint URL",
        "api_version": "Azure API version (e.g. 2024-12-01-preview)",
        "auth_type": "Auth type (api_key, client_secret, workload_identity)",
    },
    "azure_anthropic": {
        "endpoint": "Azure endpoint URL",
        "auth_type": "Auth type (api_key, client_secret, workload_identity)",
    },
    "bedrock": {
        "aws_region": "AWS region (e.g. us-east-1)",
    },
    "ollama": {},
}

VENDOR_OPTIONAL_FIELDS: Dict[str, Dict[str, str]] = {
    "openai": {
        "openai_api_base": "API base URL override",
    },
    "anthropic": {
        "anthropic_api_base": "API base URL override",
    },
    "azure_openai": {},
    "azure_anthropic": {},
    "bedrock": {
        "aws_access_key": "AWS access key",
        "aws_secret_key": "AWS secret key",
        "endpoint_url": "Custom endpoint URL",
    },
    "ollama": {},
}

AZURE_AUTH_FIELDS: Dict[str, Dict[str, str]] = {
    "api_key": {
        "api_key": "API key",
    },
    "client_secret": {
        "tenant_id": "Azure tenant ID",
        "client_id": "Azure client ID",
        "client_secret": "Client secret",
    },
    "workload_identity": {},
}

MODEL_SKIP_FIELDS = {"name"}


def __get_model_fields() -> Dict[str, str]:
    fields = {}
    if hasattr(LLMModelConfiguration, "model_fields"):
        for name, info in LLMModelConfiguration.model_fields.items():
            if name in MODEL_SKIP_FIELDS:
                continue
            desc = info.description or name.replace("_", " ").title()
            fields[name] = desc
    else:
        for name, field in LLMModelConfiguration.__fields__.items():
            if name in MODEL_SKIP_FIELDS:
                continue
            desc = field.field_info.description or name.replace("_", " ").title()
            fields[name] = desc
    return fields


def __parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def __is_sensitive(field: str) -> bool:
    return any(s in field for s in ("key", "secret", "token"))


def __mask_hint(field: str, value: str) -> str:
    if __is_sensitive(field):
        return f" [{'•' * 8}]"
    return f" [{value}]"


def __prompt_field(label: str, field: str, current: Optional[str] = None) -> str:
    hint = __mask_hint(field, current) if current else ""
    return typer.prompt(
        f"{label}{hint}",
        default=current or "",
        show_default=not __is_sensitive(field),
    )


def __prompt_vendor_config(vendor: str, existing: Dict[str, Any]) -> Dict[str, Any]:
    required = VENDOR_REQUIRED_FIELDS.get(vendor, {})
    optional = VENDOR_OPTIONAL_FIELDS.get(vendor, {})

    if not required and not optional:
        return existing

    result = dict(existing)
    console.print()
    console.print(f"  [bold]{vendor} configuration[/]")
    console.print("  [dim]Press Enter to keep current value. Type 'null' to clear.[/]")
    console.print()

    for field, description in required.items():
        current = result.get(field)
        raw = __prompt_field(f"  {field} ({description})", field, current)
        if raw:
            result[field] = raw
        elif field not in result:
            # Required fields must exist for the configuration to materialize;
            # empty is a valid value (e.g. no OpenAI organization).
            result[field] = ""

    if vendor in ("azure_openai", "azure_anthropic") and "auth_type" in result:
        auth_type = result["auth_type"]
        auth_fields = AZURE_AUTH_FIELDS.get(auth_type, {})
        if auth_fields:
            auth_cfg = result.get(auth_type, {}) or {}
            for field, description in auth_fields.items():
                current = auth_cfg.get(field)
                raw = __prompt_field(
                    f"  {auth_type}.{field} ({description})", field, current
                )
                if raw:
                    auth_cfg[field] = raw
            result[auth_type] = auth_cfg

    for field, description in optional.items():
        current = result.get(field)
        hint = f" [{current}]" if current else ""
        raw = typer.prompt(
            f"  {field} ({description}){hint}",
            default="",
            show_default=False,
        )
        if raw == "":
            continue
        if raw.lower() == "null":
            result.pop(field, None)
            continue
        result[field] = raw

    return result


model_app = typer.Typer(no_args_is_help=True)


@model_app.callback()
def model_callback():
    """
    Manage LLM model configuration.
    """


@model_app.command("configure")
def model_configure(
    model_name: Annotated[
        Optional[str],
        typer.Argument(help="Model name (e.g. gpt-4o, claude-sonnet-4)."),
    ] = None,
    vendor: Annotated[
        Optional[str],
        typer.Option("--vendor", help="Override auto-detected vendor."),
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
    Interactively configure the model, vendor credentials, and model parameters.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    identifier = resolve_agent_identifier(config, agent)

    existing_vendor, existing_model, model_cfg = config.get_model_config(identifier)

    if model_name is None:
        model_name = typer.prompt(
            "Model name", default=existing_model or "gpt-5.4-mini"
        )

    if vendor is None:
        inferred = infer_vendor(model_name)
        if existing_vendor and inferred and existing_vendor != inferred:
            vendor = typer.prompt(
                f"Current vendor is '{existing_vendor}' but model "
                f"suggests '{inferred}'. Which vendor?",
                default=existing_vendor,
            )
        elif inferred:
            vendor = inferred
        elif existing_vendor:
            vendor = existing_vendor
        else:
            vendor = typer.prompt(
                f"Could not infer vendor for '{model_name}'. Enter vendor"
            )

    existing_vendor_cfg = config.get_vendor_config(vendor, identifier)
    vendor_cfg = __prompt_vendor_config(vendor, existing_vendor_cfg)

    console.print()
    console.print("  [bold]Model parameters[/]")
    console.print("  [dim]Press Enter to keep current value. Type 'null' to clear.[/]")
    console.print()

    model_params: Dict[str, Any] = {}
    model_cfg = model_cfg or {}
    for field, description in __get_model_fields().items():
        current = model_cfg.get(field)
        hint = f" [{current}]" if current is not None else ""
        raw = typer.prompt(
            f"  {field} ({description}){hint}",
            default="",
            show_default=False,
        )
        if raw == "":
            continue
        if raw.lower() == "null":
            model_params[field] = None
            continue
        model_params[field] = __parse_value(raw)

    config.set_model(
        model_name,
        vendor,
        identifier,
        vendor_config=vendor_cfg if vendor_cfg else None,
        model_params=model_params if model_params else None,
    )
    console.print()
    console.print(f"  [green]✅ Model set to [bold]{model_name}[/] ({vendor})[/]")


@model_app.command("show")
def model_show(
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
    Show current model configuration.
    """
    project_dir = resolve_project_dir(project_dir)
    config = require_project(project_dir)
    identifier = resolve_agent_identifier(config, agent)

    vendor, model_name, model_cfg = config.get_model_config(identifier)
    console.print()
    console.print(f"  [bold]Model:[/]       [cyan]{model_name or 'not set'}[/]")
    console.print(f"  [bold]Vendor:[/]      [cyan]{vendor or 'not set'}[/]")

    if model_cfg:
        param_table = Table(show_header=True, header_style="bold")
        param_table.add_column("Parameter", style="cyan")
        param_table.add_column("Value")

        for key in __get_model_fields():
            value = model_cfg.get(key)
            if value is not None:
                param_table.add_row(key, str(value))

        console.print()
        console.print(param_table)

    vendor_cfg = config.get_vendor_config(vendor, identifier) if vendor else {}
    if vendor and vendor_cfg:
        cfg = vendor_cfg
        console.print()
        console.print(f"  [bold]{vendor} credentials[/]")
        for key, value in cfg.items():
            if isinstance(value, dict):
                console.print(f"    {key}:")
                for k, v in value.items():
                    display = "••••••••" if "key" in k or "secret" in k else str(v)
                    console.print(f"      {k}: {display}")
            else:
                display = "••••••••" if "key" in key or "secret" in key else str(value)
                console.print(f"    {key}: {display}")
    elif vendor and vendor in VENDOR_REQUIRED_FIELDS and VENDOR_REQUIRED_FIELDS[vendor]:
        console.print()
        console.print(
            f"  [yellow]⚠️  No {vendor} credentials configured."
            " Run 'model configure'.[/]"
        )

    console.print()


@model_app.command("list")
def model_list(
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    List the named LLM configurations in this project.
    """
    project_dir = resolve_any_project_dir(project_dir)
    config = ProjectConfig(project_dir)
    configs = config.list_llm_configurations()

    if not configs:
        console.print("  [dim]No LLM configurations[/]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Vendor")
    table.add_column("Model", style="dim")
    for entry in configs:
        model_name = entry.get("model_configuration", {}).get("name", "?")
        table.add_row(entry.get("name", "?"), entry.get("vendor", "?"), model_name)

    console.print()
    console.print(table)
    console.print()


def prompt_model_configuration(model_name: str) -> Dict[str, Any]:
    """Reasoning models reject function tools on chat completions
    (reasoning is on by default); they need the responses API."""
    model_type = (
        typer.prompt("Model type (chat / responses)", default="chat").strip().lower()
    )
    if model_type == "responses":
        return {"name": model_name, "type": "responses"}
    return {"name": model_name, "type": "chat", "temperature": 0.0}


@model_app.command("add")
def model_add(
    name: Annotated[str, typer.Argument(help="Name for the LLM configuration.")],
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
):
    """
    Create a named LLM configuration that agents and MCP tools can reference.
    """
    project_dir = resolve_any_project_dir(project_dir)
    config = ProjectConfig(project_dir)
    if config.get_llm_configuration(name) is not None:
        console.print(f"  [red]LLM configuration '{name}' already exists[/]")
        raise typer.Exit(code=1)

    model_name = typer.prompt("Model name", default="gpt-5.4-mini")
    model_configuration = prompt_model_configuration(model_name)
    vendor = infer_vendor(model_name) or typer.prompt("Vendor")
    vendor_cfg = __prompt_vendor_config(vendor, {})

    config.set_llm_configuration(
        name,
        vendor,
        model_configuration,
        vendor_cfg or None,
    )
    console.print()
    console.print(
        f"  [green]✅ LLM configuration '{name}' saved ({vendor} / {model_name})[/]"
    )
