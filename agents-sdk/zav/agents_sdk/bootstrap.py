from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type

from zav.llm_tracing import TracingBackendFactory
from zav.message_bus import Bootstrap, BootstrapDependency, StreamCommandHandlerRegistry

from zav.agents_sdk.adapters.event_publishers.event_publisher import (
    AbstractEventPublisher,
)

if TYPE_CHECKING:
    from zav.agents_sdk.adapters.mcp import (
        MCPOAuthIntegrationStore,
        MCPOAuthTokenClient,
    )
from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
from zav.agents_sdk.domain.chat_agent_state_store import ChatAgentStateStore
from zav.agents_sdk.domain.chat_conversation_store import ChatConversationStore
from zav.agents_sdk.handlers import CommandHandlerRegistry, EventHandlerRegistry


def setup_bootstrap(
    agent_registries_factory: AgentRegistriesFactory,
    event_publisher: Optional[AbstractEventPublisher] = None,
    tracing_backend_factory: Optional[Type[TracingBackendFactory]] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
    extra_bootstrap_deps: Optional[List[BootstrapDependency]] = None,
    command_handler_registry: Optional[Type[CommandHandlerRegistry]] = None,
    event_handler_registry: Optional[Type[EventHandlerRegistry]] = None,
    agent_state_store: Optional[ChatAgentStateStore] = None,
    chat_conversation_store: Optional[ChatConversationStore] = None,
    mcp_oauth_store: Optional[MCPOAuthIntegrationStore] = None,
    mcp_oauth_token_client: Optional[MCPOAuthTokenClient] = None,
):
    if tracing_backend_factory:
        for tracing_vendor, tracing_backend in tracing_backend_factory.registry.items():
            TracingBackendFactory.register(tracing_vendor)(tracing_backend)
    bootstrap_deps = [
        BootstrapDependency(
            name="event_publisher",
            value=event_publisher,
        ),
        BootstrapDependency(
            name="agent_registries_factory",
            value=agent_registries_factory,
        ),
        BootstrapDependency(
            name="tracing_backend_factory",
            value=TracingBackendFactory,
        ),
        BootstrapDependency(
            name="debug_backend",
            value=debug_backend,
        ),
        BootstrapDependency(
            name="agent_state_store",
            value=agent_state_store,
        ),
        BootstrapDependency(
            name="chat_conversation_store",
            value=chat_conversation_store,
        ),
    ]
    if mcp_oauth_store is not None:
        bootstrap_deps.append(
            BootstrapDependency(
                name="mcp_oauth_store",
                value=mcp_oauth_store,
            )
        )
    if mcp_oauth_token_client is not None:
        bootstrap_deps.append(
            BootstrapDependency(
                name="mcp_oauth_token_client",
                value=mcp_oauth_token_client,
            )
        )
    if extra_bootstrap_deps is not None:
        bootstrap_deps.extend(extra_bootstrap_deps)
        if command_handler_registry is not None:
            CommandHandlerRegistry.merge(command_handler_registry.registry)
        if event_handler_registry is not None:
            EventHandlerRegistry.merge(event_handler_registry.registry)
    return Bootstrap(
        dependencies=bootstrap_deps,
        command_handler_registry=CommandHandlerRegistry,
        event_handler_registry=EventHandlerRegistry,
        stream_command_handler_registry=StreamCommandHandlerRegistry,
    )
