from typing import List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_delegation.delegable_agents_source import (
    DelegableAgent,
    DelegableAgentsSource,
)
from zav.agents_sdk.adapters.agent_delegation.runnable_agents import RunnableAgents
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class InMemoryDelegableAgentsSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False, description="Enable in-memory delegable agents source."
    )
    agents: List[DelegableAgent] = Field(default_factory=list)


class InMemoryDelegableAgentsSource(DelegableAgentsSource):
    source_name = "in_memory_agents"
    source_description = "Agents listed in this configuration."

    def __init__(self, enabled: bool, agents: List[DelegableAgent]):
        self.enabled = enabled
        self.__agents = agents

    async def get_agents(self) -> List[DelegableAgent]:
        return self.__agents


class InMemoryDelegableAgentsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        runnable_agents: Optional[RunnableAgents] = None,
        in_memory_delegable_agents_source_configuration: InMemoryDelegableAgentsSourceConfiguration = InMemoryDelegableAgentsSourceConfiguration(),  # noqa: E501
    ) -> InMemoryDelegableAgentsSource:
        configuration = in_memory_delegable_agents_source_configuration
        return InMemoryDelegableAgentsSource(
            agents=[
                agent
                for agent in configuration.agents
                if runnable_agents is None
                or runnable_agents.allows(agent.agent_identifier)
            ],
            enabled=configuration.enabled,
        )
