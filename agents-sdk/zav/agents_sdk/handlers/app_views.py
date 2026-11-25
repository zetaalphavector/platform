from typing import Any, List, Optional, Type, Union

from zav.llm_tracing import TracingBackendFactory
from zav.message_bus import CommandHandlerRegistry, Message

from zav.agents_sdk.adapters.app_views import AppViewSurface
from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
from zav.agents_sdk.domain.agent_setup_retriever import require_agent_setup
from zav.agents_sdk.domain.chat_agent_factory import ChatAgentFactory
from zav.agents_sdk.domain.dependency_resolver import DependencyResolver
from zav.agents_sdk.domain.tools import AppView
from zav.agents_sdk.handlers import commands
from zav.agents_sdk.security import sanitize_bot_params


async def resolve_app_view_surface(
    cmd: Union[commands.ReadAppView, commands.CallAppTool],
    agent_registries_factory: AgentRegistriesFactory,
    tracing_backend_factory: Type[TracingBackendFactory],
) -> AppViewSurface:
    (
        agent_setup_retriever,
        chat_agent_class_registry,
        agent_dependency_registry,
        llm_configuration_store,
        secret_store,
    ) = await agent_registries_factory.create(
        tenant=cmd.tenant, request_headers=cmd.request_headers
    )
    agent_setup = await require_agent_setup(agent_setup_retriever, cmd.agent_identifier)
    # The factory only creates agents a tools source demands (delegate, task).
    chat_agent_factory = ChatAgentFactory(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=chat_agent_class_registry,
        tracing_backend_factory=tracing_backend_factory,
        trace_state_params={
            "tenant": cmd.tenant,
            **({"index_id": cmd.index_id} if cmd.index_id else {}),
            **(
                {"user_id": cmd.request_headers.requester_uuid}
                if cmd.request_headers.requester_uuid
                else {}
            ),
        },
        agent_dependency_registry=agent_dependency_registry,
        llm_configuration_store=llm_configuration_store,
        secret_store=secret_store,
    )
    resolver = DependencyResolver(
        dependency_registry=agent_dependency_registry,
        agent_creation=chat_agent_factory,
        llm_configuration_store=llm_configuration_store,
        secret_store=secret_store,
    )
    return await resolver.resolve(
        AppViewSurface,
        configuration=agent_setup.agent_configuration or {},
        handler_params={
            **sanitize_bot_params(cmd.bot_params),
            "tenant": cmd.tenant,
            "request_headers": cmd.request_headers,
            **({"index_id": cmd.index_id} if cmd.index_id else {}),
        },
        key_prefix=agent_setup.agent_identifier,
        sub_agent_mapping=agent_setup.sub_agent_mapping,
        default_llm_configuration_name=agent_setup.llm_configuration_name,
        allowed_llm_configuration_names=(
            agent_setup.allowed_llm_configuration_names or ()
        ),
    )


@CommandHandlerRegistry.register(commands.ReadAppView)
async def handle_read_app_view(
    cmd: commands.ReadAppView,
    queue: List[Message],
    agent_registries_factory: AgentRegistriesFactory,
    tracing_backend_factory: Type[TracingBackendFactory],
) -> Optional[AppView]:
    surface = await resolve_app_view_surface(
        cmd, agent_registries_factory, tracing_backend_factory
    )
    try:
        return await surface.view(cmd.view_tool)
    finally:
        await surface.cleanup()


@CommandHandlerRegistry.register(commands.CallAppTool)
async def handle_call_app_tool(
    cmd: commands.CallAppTool,
    queue: List[Message],
    agent_registries_factory: AgentRegistriesFactory,
    tracing_backend_factory: Type[TracingBackendFactory],
) -> Any:
    surface = await resolve_app_view_surface(
        cmd, agent_registries_factory, tracing_backend_factory
    )
    try:
        return await surface.call(cmd.view_tool, cmd.tool_name, cmd.arguments)
    finally:
        await surface.cleanup()
