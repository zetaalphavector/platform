from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

__all__ = ["AgentDependencyRegistry"]

try:
    from zav.agents_sdk.adapters.mcp.tools_provider import (
        MCPServerConfig,
        MCPServerTransportConfig,
        MCPToolsProvider,
        MCPToolsProviderConfiguration,
        MCPToolsProviderFactory,
        SseTransportConfig,
        StdIoTransportConfig,
    )

    AgentDependencyRegistry.register(MCPToolsProviderFactory)
    __all__ += [
        "MCPToolsProviderConfiguration",
        "MCPServerConfig",
        "MCPServerTransportConfig",
        "MCPToolsProvider",
        "MCPToolsProviderFactory",
        "StdIoTransportConfig",
        "SseTransportConfig",
    ]
except ImportError:
    pass
