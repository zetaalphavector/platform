from typing import Any, Callable, List, Optional, Type

from zav.llm_tracing import TracingBackendFactory
from zav.message_bus import Bootstrap, BootstrapDependency

from zav.agents_sdk.adapters.event_publishers.event_publisher import (
    AbstractEventPublisher,
)
from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
from zav.agents_sdk.handlers import CommandHandlerRegistry, EventHandlerRegistry


def setup_bootstrap(
    agent_registries_factory: AgentRegistriesFactory,
    event_publisher: Optional[AbstractEventPublisher] = None,
    tracing_backend_factory: Optional[Type[TracingBackendFactory]] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
    extra_bootstrap_deps: Optional[List[BootstrapDependency]] = None,
    command_handler_registry: Optional[Type[CommandHandlerRegistry]] = None,
    event_handler_registry: Optional[Type[EventHandlerRegistry]] = None,
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
    ]
    if extra_bootstrap_deps is not None:
        bootstrap_deps.extend(extra_bootstrap_deps)
    if command_handler_registry is not None:
        for command, handler in command_handler_registry.registry.items():
            CommandHandlerRegistry.register(command)(handler)
    if event_handler_registry is not None:
        for event, handlers in event_handler_registry.registry.items():
            # First check if the event is already registered, if so append the new
            # handlers
            if event in EventHandlerRegistry.registry:
                EventHandlerRegistry.registry[event].extend(handlers)
            else:
                EventHandlerRegistry.register(event)(handlers)

    return Bootstrap(
        dependencies=bootstrap_deps,
        command_handler_registry=CommandHandlerRegistry,
        event_handler_registry=EventHandlerRegistry,
    )
