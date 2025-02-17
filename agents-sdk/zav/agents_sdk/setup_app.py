from typing import Any, Callable, List, Optional, Tuple, Type

from fastapi import APIRouter, FastAPI
from zav.api import setup_api
from zav.llm_tracing import TracingBackendFactory
from zav.message_bus import (
    BootstrapDependency,
    CommandHandlerRegistry,
    EventHandlerRegistry,
)

from zav.agents_sdk.adapters.event_publishers.event_publisher import (
    AbstractEventPublisher,
)
from zav.agents_sdk.bootstrap import setup_bootstrap
from zav.agents_sdk.controllers import routers as default_routers
from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
from zav.agents_sdk.exception_handlers import add_exception_handlers


def setup_app(
    agent_registries_factory: AgentRegistriesFactory,
    event_publisher: Optional[AbstractEventPublisher] = None,
    tracing_backend_factory: Optional[Type[TracingBackendFactory]] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
    extra_bootstrap_deps: Optional[List[BootstrapDependency]] = None,
    routers: Optional[List[Tuple[str, APIRouter]]] = None,
    command_handler_registry: Optional[Type[CommandHandlerRegistry]] = None,
    event_handler_registry: Optional[Type[EventHandlerRegistry]] = None,
) -> FastAPI:
    bootstrap = setup_bootstrap(
        agent_registries_factory=agent_registries_factory,
        event_publisher=event_publisher,
        tracing_backend_factory=tracing_backend_factory,
        debug_backend=debug_backend,
        extra_bootstrap_deps=extra_bootstrap_deps,
        command_handler_registry=command_handler_registry,
        event_handler_registry=event_handler_registry,
    )

    app = FastAPI()
    add_exception_handlers(app)
    setup_api(app=app, bootstrap=bootstrap, routers=routers or default_routers)

    return app
