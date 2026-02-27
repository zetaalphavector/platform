from zav.agents_sdk.adapters.sub_agent.sub_agent_tool import (
    SubAgentConfiguration,
    SubAgentTool,
    SubAgentToolFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(SubAgentToolFactory)

__all__ = [
    "SubAgentConfiguration",
    "SubAgentTool",
    "SubAgentToolFactory",
]
