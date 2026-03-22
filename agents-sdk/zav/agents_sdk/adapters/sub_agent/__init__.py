from zav.agents_sdk.adapters.sub_agent.delegable_agents_source import (
    ConfigDelegableAgentsSource,
    ConfigDelegableAgentsSourceFactory,
    DelegableAgent,
    DelegableAgentsConfig,
    DelegableAgentsSource,
    DelegableAgentsSourceGroup,
)
from zav.agents_sdk.adapters.sub_agent.sub_agent_tool import (
    SubAgentConfiguration,
    SubAgentTool,
    SubAgentToolFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(SubAgentToolFactory)
AgentDependencyRegistry.register(ConfigDelegableAgentsSourceFactory)

__all__ = [
    "ConfigDelegableAgentsSource",
    "ConfigDelegableAgentsSourceFactory",
    "DelegableAgent",
    "DelegableAgentsConfig",
    "DelegableAgentsSource",
    "DelegableAgentsSourceGroup",
    "SubAgentConfiguration",
    "SubAgentTool",
    "SubAgentToolFactory",
]
