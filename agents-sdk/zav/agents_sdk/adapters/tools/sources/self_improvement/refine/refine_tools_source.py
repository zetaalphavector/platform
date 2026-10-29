from typing import List, Optional

from zav.pydantic_compat import Field

from zav.agents_sdk.adapters.tools.sources.self_improvement.refine.refiner import (
    Refiner,
)
from zav.agents_sdk.adapters.tools.tools_source import (
    ToolsSource,
    ToolsSourceConfiguration,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, ToolStreamingConfig

REQUEST_REFINEMENT_TOOL_NAME = "request_refinement"

_REFINE_PROMPT = """\
## Refinement

Your persistent state has three stores: memories hold durable facts, skills \
hold repeatable procedures, prompt notes hold narrow behavioural rules. When \
you notice a repeated failure or a reusable tactic, call `request_refinement` \
with a one-sentence observation; the stores are updated from it and the call \
returns what was kept. It takes effect from the next turn on, so keep working. \
Never ask for task answers to be stored.\
"""

_REQUEST_TOOL_DESCRIPTION = (
    "Refine your persistent state (memories, skills, prompt notes) from an "
    "observation about this turn: a repeated failure or a reusable tactic. "
    "Returns what was kept."
)


class RefineToolsSourceConfiguration(ToolsSourceConfiguration):
    enabled: bool = Field(
        False,
        description=(
            "Offer `request_refinement`, which runs the refiner on the agent's "
            "observation. The refiner is configured in `refiner_configuration`."
        ),
    )


class RefineToolsSource(ToolsSource):
    """The agent's side of Prime's `/refine`, as a tools source: the prompt that
    says when to ask, and the `request_refinement` tool that runs the refiner
    on the agent's observation, right there, like any other tool call."""

    source_name = "refine_tools"
    source_description = "Let the agent review and refine its own work."
    source_gate = "attempt_lineage"

    def __init__(
        self, refiner: Refiner, enabled: bool, defer_tools: Optional[bool] = None
    ) -> None:
        self.enabled = enabled and bool(refiner.kinds)
        self.defer_tools = defer_tools
        self.__refiner = refiner

    async def to_prompt(self) -> str:
        return _REFINE_PROMPT if self.enabled else ""

    async def get_tools(self) -> List[Tool]:
        if not self.enabled:
            return []

        async def request_refinement(observation: Optional[str] = None) -> str:
            return await self.__refiner.refine(
                focus=(observation or "").strip(), transcript=None, mode="requested"
            )

        return [
            Tool(
                name=REQUEST_REFINEMENT_TOOL_NAME,
                description=_REQUEST_TOOL_DESCRIPTION,
                executable=request_refinement,
                parameters_spec={
                    "type": "object",
                    "properties": {
                        "observation": {
                            "type": "string",
                            "description": (
                                "What you observed and what should be persisted."
                            ),
                        }
                    },
                    "required": ["observation"],
                },
                streaming_config=ToolStreamingConfig(
                    running_text="Refining...",
                    completed_text="Refined",
                ),
            )
        ]


class RefineToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        refiner: Refiner,
        refine_tools_source_configuration: RefineToolsSourceConfiguration = (
            RefineToolsSourceConfiguration()
        ),
    ) -> RefineToolsSource:
        return RefineToolsSource(
            refiner=refiner,
            enabled=refine_tools_source_configuration.enabled,
            defer_tools=refine_tools_source_configuration.defer_tools,
        )
