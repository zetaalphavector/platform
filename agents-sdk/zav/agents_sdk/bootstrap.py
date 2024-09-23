from typing import Any, Callable, Optional, Type

from zav.llm_tracing import TracingBackendFactory
from zav.message_bus import Bootstrap, BootstrapDependency

from zav.agents_sdk.adapters import AgentDependencyRegistry
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetupRetriever
from zav.agents_sdk.domain.chat_agent_factory import ChatAgentFactory
from zav.agents_sdk.handlers import CommandHandlerRegistry, EventHandlerRegistry


def setup_bootstrap(
    agent_setup_retriever: AgentSetupRetriever,
    chat_agent_factory: Type[ChatAgentFactory],
    tracing_backend_factory: Optional[Type[TracingBackendFactory]] = None,
    agent_dependency_registry: Optional[Type[AgentDependencyRegistry]] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
):
    if agent_dependency_registry:
        for dep in agent_dependency_registry.registry.values():
            AgentDependencyRegistry.register(dep)
    if tracing_backend_factory:
        for tracing_vendor, tracing_backend in tracing_backend_factory.registry.items():
            TracingBackendFactory.register(tracing_vendor)(tracing_backend)
    bootstrap_deps = [
        BootstrapDependency(
            name="agent_setup_retriever",
            value=agent_setup_retriever,
        ),
        BootstrapDependency(
            name="chat_agent_factory",
            value=chat_agent_factory,
        ),
        BootstrapDependency(
            name="tracing_backend_factory",
            value=TracingBackendFactory,
        ),
        BootstrapDependency(
            name="agent_dependency_registry",
            value=AgentDependencyRegistry,
        ),
        BootstrapDependency(
            name="debug_backend",
            value=debug_backend,
        ),
    ]
    return Bootstrap(
        dependencies=bootstrap_deps,
        command_handler_registry=CommandHandlerRegistry,
        event_handler_registry=EventHandlerRegistry,
    )
