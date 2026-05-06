import asyncio
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from typing_extensions import Annotated

from zav.agents_sdk.cli.project_config import CONFIG_FILE, ENV_DIR
from zav.agents_sdk.cli.require import resolve_project_dir

console = Console()


async def run_agent_headless(
    prompt: str,
    project_dir: str,
    agent_identifier: str = "agent",
    setup_src: Optional[str] = None,
    secret_setup_src: Optional[str] = None,
) -> Optional[str]:
    # Deferred imports: SDK domain/adapter modules pull in heavy internals —
    # importing at module level adds seconds to CLI startup for commands that
    # don't need them.
    from zav.llm_tracing import TracingBackendFactory

    from zav.agents_sdk import AgentDependencyRegistry, AgentSetupRetrieverFromFile
    from zav.agents_sdk.domain.chat_agent_factory import ChatAgentFactory
    from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistry
    from zav.agents_sdk.domain.chat_message import ChatMessage, ChatMessageSender

    if setup_src is None:
        setup_src = os.path.join(project_dir, CONFIG_FILE)
    if secret_setup_src is None:
        secret_setup_src = os.path.join(project_dir, ENV_DIR, CONFIG_FILE)

    agent_setup_retriever = AgentSetupRetrieverFromFile(
        file_path=setup_src, secret_file_path=secret_setup_src
    )

    chat_agent_factory = ChatAgentFactory(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=ChatAgentClassRegistry,
        tracing_backend_factory=TracingBackendFactory,
        trace_state_params={},
        agent_dependency_registry=AgentDependencyRegistry,
    )

    agent = await chat_agent_factory.create(
        agent_identifier=agent_identifier,
        handler_params={"request_headers": {}},
    )

    conversation = [
        ChatMessage(sender=ChatMessageSender.USER, content=prompt),
    ]

    original_cwd = os.getcwd()
    os.chdir(project_dir)
    try:
        response = await agent.execute(conversation)
    finally:
        os.chdir(original_cwd)

    if response is None:
        return None
    return response.content


def run_command(
    prompt: Annotated[
        str,
        typer.Argument(help="The prompt to send to the agent."),
    ],
    project_dir: Annotated[
        Optional[str],
        typer.Option("--project-dir", help="Project directory."),
    ] = None,
    agent_identifier: Annotated[
        str,
        typer.Option("--agent", help="Agent identifier to use."),
    ] = "agent",
):
    """
    Run the agent headlessly with a single prompt.

    Sends the prompt, waits for the agent to finish (including any tool
    calls), and prints the response.
    """
    project_dir = resolve_project_dir(project_dir)
    project_dir = os.path.abspath(project_dir)

    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    import zav.agents_sdk.agents  # noqa: F401

    if os.path.isfile(os.path.join(project_dir, "__init__.py")):
        from zav.agents_sdk.cli.load_chat_agent_factory import (
            from_string as import_chat_agent_class_registry_from_string,
        )

        import_chat_agent_class_registry_from_string(project_dir)

    try:
        response = asyncio.run(
            run_agent_headless(prompt, project_dir, agent_identifier)
        )
    except Exception as e:
        console.print(f"[red]Agent error: {e}[/]")
        raise typer.Exit(code=1)

    if response is None:
        console.print("[yellow]Agent returned no response.[/]")
        raise typer.Exit(code=1)

    console.print()
    console.print(Panel(Markdown(response), border_style="green"))
    console.print()
