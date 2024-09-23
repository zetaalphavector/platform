from typing import Any, Callable, Optional, Type

from fastapi import FastAPI
from zav.api import setup_api
from zav.llm_tracing import TracingBackendFactory

from zav.agents_sdk.bootstrap import setup_bootstrap
from zav.agents_sdk.controllers import routers
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetupRetriever
from zav.agents_sdk.domain.chat_agent_factory import ChatAgentFactory
from zav.agents_sdk.exception_handlers import add_exception_handlers


def setup_app(
    agent_setup_retriever: AgentSetupRetriever,
    chat_agent_factory: Type[ChatAgentFactory],
    tracing_backend_factory: Optional[Type[TracingBackendFactory]] = None,
    agent_dependency_registry: Optional[Type[AgentDependencyRegistry]] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
) -> FastAPI:
    bootstrap = setup_bootstrap(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_factory=chat_agent_factory,
        tracing_backend_factory=tracing_backend_factory,
        agent_dependency_registry=agent_dependency_registry,
        debug_backend=debug_backend,
    )

    app = FastAPI()

    add_exception_handlers(app)
    setup_api(app=app, bootstrap=bootstrap, routers=routers)
    return app
