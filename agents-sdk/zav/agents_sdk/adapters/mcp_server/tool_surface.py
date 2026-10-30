import asyncio
from typing import Any, Dict, List, Optional

from zav.agents_sdk.adapters.memory.memory_provider import MemoryProvider
from zav.agents_sdk.adapters.skills.skills_provider import SkillsProvider
from zav.agents_sdk.adapters.tool_policy import ToolPolicyProvider
from zav.agents_sdk.adapters.tools.tools_provider import ToolsProvider
from zav.agents_sdk.domain.agent_dependency import (
    AgentDependencyFactory,
    AgentDependencyRegistry,
)
from zav.agents_sdk.domain.tools import Tool, ToolsRegistry

_RULE_ASKS_MESSAGE = (
    "A rule asks the user before this {name} call, which an MCP call cannot do."
)


class ToolSurface:
    """The externally exposable tool surface: every capability provider whose
    tools are safe to serve outside the agent loop. The MCP exposure serves
    exactly this; the agent merges it with its loop-only providers (delegation,
    consume-side MCP, context). A capability joins by becoming a
    ToolSurfaceFactory.create parameter — deliberately, never implicitly.
    The tool rules apply as in the agent loop, on each call and its result
    too; a tool that asks first is left out, since an MCP call cannot pause
    for the user."""

    def __init__(
        self,
        tools_provider: ToolsProvider,
        memory_provider: MemoryProvider,
        skills_provider: SkillsProvider,
        tool_policy: ToolPolicyProvider,
    ):
        self.tools_provider = tools_provider
        self.memory_provider = memory_provider
        self.skills_provider = skills_provider
        self.__tool_policy = tool_policy

    async def get_tools(self) -> List[Tool]:
        collected = await asyncio.gather(
            self.tools_provider.get_tools(),
            self.memory_provider.get_tools(),
            self.skills_provider.get_tools(),
        )
        # No conversation: rules that only shape what a model sees step aside.
        offered = await self.__tool_policy.apply(
            [tool for tools in collected for tool in tools], None
        )
        return [
            tool
            for tool in offered.tools
            if tool.options.confirmation not in ("ask", "require")
        ]

    async def call(self, name: str, params: Optional[Dict[str, Any]]) -> Any:
        tools_registry = ToolsRegistry()
        tools_registry.extend(await self.get_tools())
        return await self.__tool_policy.execute_outside_turn(
            tools_registry, name, params, _RULE_ASKS_MESSAGE.format(name=name)
        )


class ToolSurfaceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        tools_provider: ToolsProvider,
        memory_provider: MemoryProvider,
        skills_provider: SkillsProvider,
        tool_policy_provider: ToolPolicyProvider,
    ) -> ToolSurface:
        return ToolSurface(
            tools_provider=tools_provider,
            memory_provider=memory_provider,
            skills_provider=skills_provider,
            tool_policy=tool_policy_provider,
        )


AgentDependencyRegistry.register(ToolSurfaceFactory)
