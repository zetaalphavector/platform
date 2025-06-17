import importlib.util

from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

__all__ = ["AgentDependencyRegistry"]

if importlib.util.find_spec("mcp") is not None:
    from zav.agents_sdk.adapters.mcp.tools_provider import (
        MCPConfiguration,
        MCPServerConfig,
        MCPServerTransportConfig,
        MCPToolsProvider,
        MCPToolsProviderFactory,
        SseTransportConfig,
        StdIoTransportConfig,
    )

    AgentDependencyRegistry.register(MCPToolsProviderFactory)
    __all__ += [
        "MCPConfiguration",
        "MCPServerConfig",
        "MCPServerTransportConfig",
        "MCPToolsProvider",
        "MCPToolsProviderFactory",
        "StdIoTransportConfig",
        "SseTransportConfig",
    ]
