from __future__ import annotations

from contextlib import asynccontextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

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

if TYPE_CHECKING:
    from zav.agents_sdk.adapters.mcp import (
        MCPOAuthIntegrationStore,
        MCPOAuthTokenClient,
    )
from zav.agents_sdk.adapters.stream_buffer import StreamBufferRegistry
from zav.agents_sdk.bootstrap import setup_bootstrap
from zav.agents_sdk.controllers import routers as default_routers
from zav.agents_sdk.dependencies import (
    get_chat_stream_recording_store,
    get_chat_stream_supervision_store,
    get_stream_buffer_registry,
)
from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
from zav.agents_sdk.domain.chat_agent_state_store import ChatAgentStateStore
from zav.agents_sdk.domain.chat_conversation_store import ChatConversationStore
from zav.agents_sdk.domain.chat_stream_recording_store import ChatStreamRecordingStore
from zav.agents_sdk.domain.chat_stream_supervision_store import (
    ChatStreamSupervisionStore,
)
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
    stream_buffer_registry: Optional[StreamBufferRegistry] = None,
    agent_state_store: Optional[ChatAgentStateStore] = None,
    chat_stream_recording_store: Optional[ChatStreamRecordingStore] = None,
    chat_stream_supervision_store: Optional[ChatStreamSupervisionStore] = None,
    chat_conversation_store: Optional[ChatConversationStore] = None,
    mcp_oauth_store: Optional[MCPOAuthIntegrationStore] = None,
    mcp_oauth_token_client: Optional[MCPOAuthTokenClient] = None,
) -> FastAPI:
    if stream_buffer_registry is None:
        stream_buffer_registry = StreamBufferRegistry()

    @asynccontextmanager
    async def stream_buffer_registry_lifespan(
        app: FastAPI,
    ) -> AsyncIterator[Dict[str, object]]:
        yield {
            **get_stream_buffer_registry.bind(stream_buffer_registry),
            **get_chat_stream_recording_store.bind(chat_stream_recording_store),
            **get_chat_stream_supervision_store.bind(chat_stream_supervision_store),
        }

    bootstrap = setup_bootstrap(
        agent_registries_factory=agent_registries_factory,
        event_publisher=event_publisher,
        tracing_backend_factory=tracing_backend_factory,
        debug_backend=debug_backend,
        extra_bootstrap_deps=extra_bootstrap_deps,
        command_handler_registry=command_handler_registry,
        event_handler_registry=event_handler_registry,
        agent_state_store=agent_state_store,
        chat_conversation_store=chat_conversation_store,
        mcp_oauth_store=mcp_oauth_store,
        mcp_oauth_token_client=mcp_oauth_token_client,
    )

    app = setup_api(
        bootstrap=bootstrap,
        routers=routers or default_routers,
        lifespans=[stream_buffer_registry_lifespan],
    )
    add_exception_handlers(app)

    return app
