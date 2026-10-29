from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.agent_creator import SubAgentApprovals
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
    sub_agent_approvals: Optional[SubAgentApprovals] = Field(
        None,
        description=(
            "What this agent does with a call that needs the user's approval: "
            "ask the user in this chat, decline it, or run it. It can only be "
            "stricter than the provider's sub_agent_approvals."
        ),
    )


class DelegableAgentsSource(ABC):
    """Abstract source for discovering agents available for delegation.

    Extend this to provide delegable agents from any origin — for example,
    from a static configuration list, from the user-agents API, or from
    tenant settings.
    """

    source_name: ClassVar[str]
    # One line for admins choosing sources, e.g. in the agent editor.
    source_description: ClassVar[Optional[str]] = None
    # The tenant gate that hides this source until a tenant enables it.
    source_gate: ClassVar[Optional[str]] = None
    enabled: bool

    @abstractmethod
    async def get_agents(self) -> List[DelegableAgent]:
        """Return the agents available for delegation from this source."""
        raise NotImplementedError


class WritableDelegableAgentsSource(DelegableAgentsSource):
    """A source the harness can add sub-agents to. Only sources of this type
    qualify as the delegation provider's ``writable`` source."""

    @property
    @abstractmethod
    def available_tools(self) -> List[str]:
        """Tool source names a new sub-agent may be given."""
        raise NotImplementedError

    @abstractmethod
    async def create_agent(
        self, name: str, description: str, instructions: str, tools: List[str]
    ) -> None:
        """Create or replace the named sub-agent with `instructions` as its
        system prompt and `tools` as the tool sources it may use."""
        raise NotImplementedError


class DelegableAgentsSourceGroup(DependencyGroup[DelegableAgentsSource]):
    """Collects all registered ``DelegableAgentsSource`` instances."""

    __collects__ = DelegableAgentsSource
