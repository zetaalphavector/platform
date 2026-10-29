from zav.agents_sdk.adapters.tool_search.sources import (
    ClientToolSearchSource,
    ClientToolSearchSourceConfiguration,
    ClientToolSearchSourceFactory,
    EmulatedToolSearchSource,
    EmulatedToolSearchSourceConfiguration,
    EmulatedToolSearchSourceFactory,
    ServerToolSearchSource,
    ServerToolSearchSourceConfiguration,
    ServerToolSearchSourceFactory,
)
from zav.agents_sdk.adapters.tool_search.tool_search_provider import (
    ToolSearchProvider,
    ToolSearchProviderConfiguration,
    ToolSearchProviderFactory,
)
from zav.agents_sdk.adapters.tool_search.tool_search_source import (
    ToolSearchSource,
    ToolSearchSourceGroup,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(ToolSearchProviderFactory)
# Registration order is the default preference order.
AgentDependencyRegistry.register(ServerToolSearchSourceFactory)
AgentDependencyRegistry.register(ClientToolSearchSourceFactory)
AgentDependencyRegistry.register(EmulatedToolSearchSourceFactory)

__all__ = [
    "ClientToolSearchSource",
    "ClientToolSearchSourceConfiguration",
    "ClientToolSearchSourceFactory",
    "EmulatedToolSearchSource",
    "EmulatedToolSearchSourceConfiguration",
    "EmulatedToolSearchSourceFactory",
    "ServerToolSearchSource",
    "ServerToolSearchSourceConfiguration",
    "ServerToolSearchSourceFactory",
    "ToolSearchProvider",
    "ToolSearchProviderConfiguration",
    "ToolSearchProviderFactory",
    "ToolSearchSource",
    "ToolSearchSourceGroup",
]
