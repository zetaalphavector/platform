from dataclasses import dataclass
from typing import Collection, Dict, List, Optional

from zav.agents_sdk.adapters.llm_models.chat_completion_types import ChatCompletion
from zav.agents_sdk.adapters.tool_policy.tool_policy_source import (
    ToolPolicyConfiguration,
    ToolPolicySource,
    ToolPolicySourceGroup,
    active_rule_sources,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool


@dataclass(frozen=True)
class ToolSelection:
    tools: List[Tool]
    # The source that removed each tool that was removed.
    removed_by: Dict[str, str]


class ToolPolicyProvider:
    """Runs the rule sources one after the other, each on what the one before
    it kept and set. No source waives a required ask."""

    def __init__(self, sources: List[ToolPolicySource]):
        self.__sources = sources

    async def apply(
        self,
        tools: List[Tool],
        transcript: Optional[List[ChatCompletion]],
        excluding: Collection[str] = (),
    ) -> ToolSelection:
        removed_by: Dict[str, str] = {}
        for source in self.__sources:
            if source.source_name in excluding:
                continue
            kept = {tool.name: tool for tool in await source.apply(tools, transcript)}
            for tool in tools:
                if tool.name not in kept:
                    removed_by[tool.name] = source.source_name
            tools = [
                self.__keep_required(tool, kept[tool.name])
                for tool in tools
                if tool.name in kept
            ]
        return ToolSelection(tools=tools, removed_by=removed_by)

    @staticmethod
    def __keep_required(before: Tool, after: Tool) -> Tool:
        if (
            before.options.confirmation != "require"
            or after.options.confirmation == "require"
        ):
            return after
        return after.copy(
            update={
                "options": after.options.copy(
                    update={
                        "confirmation": "require",
                        "confirmation_source": before.options.confirmation_source,
                    }
                )
            }
        )


class ToolPolicyProviderFactory(AgentDependencyFactory):
    # One per agent: its turn, its REPL and its MCP exposure apply alike.
    __singleton__ = True

    @classmethod
    def create(
        cls,
        tool_policy_source_group: ToolPolicySourceGroup = ToolPolicySourceGroup(
            items=[]
        ),
        tool_policy_configuration: ToolPolicyConfiguration = (
            ToolPolicyConfiguration()
        ),
    ) -> ToolPolicyProvider:
        return ToolPolicyProvider(
            active_rule_sources(
                tool_policy_source_group.items, tool_policy_configuration
            )
        )
