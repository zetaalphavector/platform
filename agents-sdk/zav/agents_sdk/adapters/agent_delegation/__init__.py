from zav.agents_sdk.adapters.agent_delegation.agent_delegation_provider import (
    AgentDelegationProvider,
    AgentDelegationProviderConfiguration,
    AgentDelegationProviderFactory,
)
from zav.agents_sdk.adapters.agent_delegation.delegable_agents_source import (
    DelegableAgent,
    DelegableAgentsSource,
    DelegableAgentsSourceGroup,
)
from zav.agents_sdk.adapters.agent_delegation.sources.in_memory_agents_source import (
    InMemoryDelegableAgentsSource,
    InMemoryDelegableAgentsSourceConfiguration,
    InMemoryDelegableAgentsSourceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(AgentDelegationProviderFactory)
AgentDependencyRegistry.register(InMemoryDelegableAgentsSourceFactory)

__all__ = [
    "InMemoryDelegableAgentsSource",
    "InMemoryDelegableAgentsSourceConfiguration",
    "InMemoryDelegableAgentsSourceFactory",
    "DelegableAgent",
    "DelegableAgentsSource",
    "DelegableAgentsSourceGroup",
    "AgentDelegationProvider",
    "AgentDelegationProviderConfiguration",
    "AgentDelegationProviderFactory",
]
