from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.agent_creator import SubAgentCount
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class SubAgentLimitsConfiguration(BaseModel):
    """Limits on the agents a turn starts with ``task`` and ``delegate``."""

    max_depth: int = Field(
        1,
        description=(
            "Depth below the root agent at which spawning stops: an agent this "
            "deep gets no task or delegate tool. At 1, agents started by either "
            "tool get neither."
        ),
    )
    max_sub_agents: int = Field(
        8,
        description=(
            "Agents the whole turn may start with task and delegate together, "
            "counted across every depth; once the turn has started this many, "
            "this agent's further calls are refused."
        ),
    )


class SubAgentLimits:
    """One agent's limits, applied to the turn's shared ``SubAgentCount``."""

    def __init__(
        self,
        sub_agent_count: SubAgentCount,
        agent_depth: int,
        max_depth: int,
        max_sub_agents: int,
    ):
        self.can_spawn = agent_depth < max_depth and max_sub_agents > 0
        # A task sub-agent runs this same setup one level deeper.
        self.task_sub_agent_can_spawn = (
            agent_depth + 1 < max_depth and max_sub_agents > 0
        )
        self.__sub_agent_count = sub_agent_count
        self.__max_sub_agents = max_sub_agents

    def try_start(self) -> bool:
        # Synchronous, so parallel tool calls cannot all pass the check.
        if self.__sub_agent_count.started >= self.__max_sub_agents:
            return False
        self.__sub_agent_count.started += 1
        return True

    @property
    def refusal(self) -> str:
        return (
            f"Sub-agent limit reached: {self.__max_sub_agents} agents have already "
            "been started this turn with `task` and `delegate`. Do not call either "
            "again; do the remaining work yourself."
        )


class SubAgentLimitsFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(
        cls,
        sub_agent_count: SubAgentCount,
        sub_agent_limits_configuration: SubAgentLimitsConfiguration = (
            SubAgentLimitsConfiguration()
        ),
        agent_depth: int = 0,
    ) -> SubAgentLimits:
        return SubAgentLimits(
            sub_agent_count=sub_agent_count,
            agent_depth=agent_depth,
            max_depth=sub_agent_limits_configuration.max_depth,
            max_sub_agents=sub_agent_limits_configuration.max_sub_agents,
        )
