from abc import ABC, abstractmethod
from typing import ClassVar, List

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.agent_dependency import (
    AgentDependencyFactory,
    DependencyGroup,
)


class DelegableAgent(BaseModel):
    """An agent that can be targeted by the delegate tool."""

    agent_identifier: str = Field(
        description="The agent_identifier used to create the agent.",
    )
    name: str = Field(
        description="Short display name shown to the LLM in the enum.",
    )
    description: str = Field(
        description="One-line description of what this agent specializes in.",
    )


class DelegableAgentsSource(ABC):
    """Abstract source for discovering agents available for delegation.

    Extend this to provide delegable agents from any origin — for example,
    from a static configuration list, from the user-agents API, or from
    tenant settings.
    """

    source_name: ClassVar[str]

    @abstractmethod
    async def get_agents(self) -> List[DelegableAgent]:
        """Return the agents available for delegation from this source."""
        raise NotImplementedError


class DelegableAgentsSourceGroup(DependencyGroup[DelegableAgentsSource]):
    """Collects all registered ``DelegableAgentsSource`` instances."""

    __collects__ = DelegableAgentsSource


class DelegableAgentsConfig(BaseModel):
    agents: List[DelegableAgent] = Field(default_factory=list)


class ConfigDelegableAgentsSource(DelegableAgentsSource):
    source_name = "config"

    def __init__(self, agents: List[DelegableAgent]):
        self.__agents = agents

    async def get_agents(self) -> List[DelegableAgent]:
        return self.__agents


class ConfigDelegableAgentsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        delegable_agents_config: DelegableAgentsConfig = DelegableAgentsConfig(),
    ) -> ConfigDelegableAgentsSource:
        return ConfigDelegableAgentsSource(
            agents=delegable_agents_config.agents,
        )
