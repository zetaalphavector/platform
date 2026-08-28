from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.agent_dependency import DependencyGroup


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
    bot_params: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Runtime bot_params for this agent. When present, these are "
            "passed to the agent factory so the delegated agent runs with "
            "the same configuration as a direct invocation."
        ),
    )
    allow_llm_selection: Optional[bool] = Field(
        None,
        description=(
            "Whether the delegating agent may pick this agent's LLM by name. "
            "Unset falls back to the provider's allow_llm_selection."
        ),
    )


class DelegableAgentsSource(ABC):
    """Abstract source for discovering agents available for delegation.

    Extend this to provide delegable agents from any origin — for example,
    from a static configuration list, from the user-agents API, or from
    tenant settings.
    """

    source_name: ClassVar[str]
    enabled: bool

    @abstractmethod
    async def get_agents(self) -> List[DelegableAgent]:
        """Return the agents available for delegation from this source."""
        raise NotImplementedError


class DelegableAgentsSourceGroup(DependencyGroup[DelegableAgentsSource]):
    """Collects all registered ``DelegableAgentsSource`` instances."""

    __collects__ = DelegableAgentsSource
