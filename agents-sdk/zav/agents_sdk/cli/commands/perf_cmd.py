import os
import sys
from typing import Dict, List, Optional

import typer
import uvicorn
from rich.console import Console
from typing_extensions import Annotated

from zav.agents_sdk.cli.require import resolve_project_dir

console = Console()
perf_app = typer.Typer(no_args_is_help=True)

DEFAULT_SERVERS = [
    "server-1=http://127.0.0.1:8001",
    "server-2=http://127.0.0.1:8002",
]
DEFAULT_PROMPT = (
    "Write a detailed, multi-paragraph explanation of how TCP congestion "
    "control works, covering slow start, congestion avoidance, fast retransmit "
    "and fast recovery. Take your time and be thorough."
)


def __parse_servers(values: List[str]) -> Dict[str, str]:
    servers: Dict[str, str] = {}
    for i, value in enumerate(values, start=1):
        if "=" in value:
            label, url = value.split("=", 1)
        else:
            label, url = f"server-{i}", value
        servers[label.strip()] = url.strip().rstrip("/")
    return servers


@perf_app.command("server")
def perf_server(
    project_dir: Annotated[
        Optional[str],
        typer.Argument(help="The project directory where the agents are located."),
    ] = None,
    port: Annotated[
        int,
        typer.Option(help="Port to listen on (use a distinct port per server)."),
    ] = 8001,
    label: Annotated[
        str,
        typer.Option(help="Server name reported to the perf client."),
    ] = "server-1",
    storage_path: Annotated[
        str,
        typer.Option(
            help="Shared state directory. All servers MUST point at the same "
            "path so a turn checkpointed by one server can be resumed by another."
        ),
    ] = ".perf-storage",
    setup_src: Annotated[
        Optional[str],
        typer.Option(help="Path of the agent setup configuration file."),
    ] = None,
    secret_setup_src: Annotated[
        Optional[str],
        typer.Option(help="Path of the secret agent setup configuration file."),
    ] = None,
    host: Annotated[
        str,
        typer.Option(help="Host to listen on."),
    ] = "127.0.0.1",
    reload: Annotated[
        bool,
        typer.Option(help="Enable auto-reload."),
    ] = False,
):
    """
    Run one perf server: a real REST API process with the resumable agent
    endpoints and a gated self-destruct route for crash testing.
    """
    project_dir = os.path.abspath(resolve_project_dir(project_dir))
    storage_path = os.path.abspath(storage_path)
    if setup_src is None:
        setup_src = os.path.join(project_dir, "agent_setups.json")
    if secret_setup_src is None:
        secret_setup_src = os.path.join(project_dir, "env", "agent_setups.json")

    os.environ["JSON_LOGGING"] = "0"
    os.environ["STORAGE_BACKEND"] = "disk"
    os.environ["STORAGE_PATH"] = storage_path
    os.environ["ZAV_PROJECT_DIR"] = project_dir
    os.environ["ZAV_AGENT_SETUP_SRC"] = setup_src
    os.environ["ZAV_SECRET_AGENT_SETUP_SRC"] = secret_setup_src
    os.environ["ZAV_PERF_MODE"] = "1"
    os.environ["ZAV_PERF_LABEL"] = label
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [project_dir, os.getenv("PYTHONPATH")])
    )

    sys.path.insert(0, project_dir)
    os.chdir(project_dir)

    console.print(
        f"  [bold cyan]{label}[/] on [bold]{host}:{port}[/] "
        f"sharing state at [bold]{storage_path}[/]"
    )
    uvicorn.run(
        "zav.agents_sdk.cli.perf_app:app",
        host=host,
        port=port,
        reload=reload,
    )


@perf_app.command("client")
def perf_client(
    server: Annotated[
        Optional[List[str]],
        typer.Option(
            "--server",
            help="Server as 'name=url' (repeatable). Defaults to two local servers.",
        ),
    ] = None,
    tenant: Annotated[
        str,
        typer.Option(help="Tenant to run the turn under."),
    ] = "zetaalpha",
    agent: Annotated[
        str,
        typer.Option(help="Agent identifier to run."),
    ] = "agent",
    prompt: Annotated[
        str,
        typer.Option(help="Prompt for the turn (pick one with a long answer)."),
    ] = DEFAULT_PROMPT,
    message_id: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Attach to an existing turn's stream. Pass a copied handle "
                "'session_id:message_id' to attach across pods; a bare message "
                "id attaches same-pod only."
            )
        ),
    ] = None,
):
    """
    Stream a turn from a perf server and measure it live: time to first message,
    inter-message gaps, and recovery after a server crash (re-attach or resume).
    """
    # Deferred: the client pulls in httpx + a raw-terminal TUI that the rest of
    # the CLI never needs, so importing it here keeps startup snappy.
    from zav.agents_sdk.cli.perf_client import run_client

    servers = __parse_servers(server or DEFAULT_SERVERS)
    run_client(
        servers=servers,
        tenant=tenant,
        agent_identifier=agent,
        prompt=prompt,
        message_id=message_id,
    )
