from typing import Any, List, Optional, Type, Union

from zav.llm_tracing import TracingBackendFactory
from zav.message_bus import CommandHandlerRegistry, Message

from zav.agents_sdk.adapters.mcp_server.tool_surface import ToolSurface
from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
from zav.agents_sdk.domain.chat_agent_factory import ChatAgentFactory
from zav.agents_sdk.domain.dependency_resolver import DependencyResolver
from zav.agents_sdk.domain.mcp_server_setup import (
    MCPServerSetupNotFound,
    MCPServerSetupRetriever,
)
from zav.agents_sdk.domain.tools import Tool, ToolsRegistry
from zav.agents_sdk.handlers import commands


async def resolve_mcp_tool_surface(
    cmd: Union[commands.ListMCPTools, commands.CallMCPTool],
    agent_registries_factory: AgentRegistriesFactory,
    tracing_backend_factory: Type[TracingBackendFactory],
    mcp_server_setup_retriever: Optional[MCPServerSetupRetriever],
) -> ToolSurface:
    if mcp_server_setup_retriever is None:
        raise MCPServerSetupNotFound("MCP server setups are not configured.")
    if not cmd.mcp_server_identifier:
        raise MCPServerSetupNotFound(
            "Missing required 'mcp_server_identifier' query parameter."
        )
    setup = await mcp_server_setup_retriever.get(cmd.tenant, cmd.mcp_server_identifier)
    if setup is None:
        raise MCPServerSetupNotFound(
            f"Unknown MCP server setup '{cmd.mcp_server_identifier}'."
        )
    (
        agent_setup_retriever,
        chat_agent_class_registry,
        agent_dependency_registry,
        llm_configuration_store,
    ) = await agent_registries_factory.create(tenant=cmd.tenant)
    # The factory participates only in its true role: creating agents when a
    # tool source in the bundle demands one (e.g. the delegate/task tools).
    trace_state_params = {
        "tenant": cmd.tenant,
        **({"index_id": cmd.index_id} if cmd.index_id else {}),
        **(
            {"user_id": cmd.request_headers.requester_uuid}
            if cmd.request_headers.requester_uuid
            else {}
        ),
    }
    chat_agent_factory = ChatAgentFactory(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=chat_agent_class_registry,
        tracing_backend_factory=tracing_backend_factory,
        trace_state_params=trace_state_params,
        agent_dependency_registry=agent_dependency_registry,
        llm_configuration_store=llm_configuration_store,
    )
    # A bundle has no agent setup: the exposure's llm_selection_configuration
    # or the caller supplies the name; requests are unrestricted within the
    # tenant's store.
    resolver = DependencyResolver(
        dependency_registry=agent_dependency_registry,
        agent_creation=chat_agent_factory,
        llm_configuration_store=llm_configuration_store,
    )
    return await resolver.resolve(
        ToolSurface,
        configuration=setup.configuration or {},
        handler_params={
            "tenant": cmd.tenant,
            "request_headers": cmd.request_headers,
            "index_id": cmd.index_id,
            **(
                {
                    "llm_selection_configuration": {
                        "llm_configuration_name": cmd.llm_configuration_name
                    }
                }
                if cmd.llm_configuration_name
                else {}
            ),
        },
        key_prefix=setup.mcp_server_identifier,
    )


@CommandHandlerRegistry.register(commands.ListMCPTools)
async def handle_list_mcp_tools(
    cmd: commands.ListMCPTools,
    queue: List[Message],
    agent_registries_factory: AgentRegistriesFactory,
    tracing_backend_factory: Type[TracingBackendFactory],
    mcp_server_setup_retriever: Optional[MCPServerSetupRetriever] = None,
) -> List[Tool]:
    tool_surface = await resolve_mcp_tool_surface(
        cmd,
        agent_registries_factory,
        tracing_backend_factory,
        mcp_server_setup_retriever,
    )
    return await tool_surface.get_tools()


@CommandHandlerRegistry.register(commands.CallMCPTool)
async def handle_call_mcp_tool(
    cmd: commands.CallMCPTool,
    queue: List[Message],
    agent_registries_factory: AgentRegistriesFactory,
    tracing_backend_factory: Type[TracingBackendFactory],
    mcp_server_setup_retriever: Optional[MCPServerSetupRetriever] = None,
) -> Any:
    tool_surface = await resolve_mcp_tool_surface(
        cmd,
        agent_registries_factory,
        tracing_backend_factory,
        mcp_server_setup_retriever,
    )
    tools_registry = ToolsRegistry()
    tools_registry.extend(await tool_surface.get_tools())
    return await tools_registry.execute(name=cmd.tool_name, params=cmd.arguments)
