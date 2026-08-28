import asyncio
from typing import List

from zav.agents_sdk.adapters.memory.memory_provider import MemoryProvider
from zav.agents_sdk.adapters.skills.skills_provider import SkillsProvider
from zav.agents_sdk.adapters.tools.tools_provider import ToolsProvider
from zav.agents_sdk.domain.agent_dependency import (
    AgentDependencyFactory,
    AgentDependencyRegistry,
)
from zav.agents_sdk.domain.tools import Tool


class ToolSurface:
    """The externally exposable tool surface: every capability provider whose
    tools are safe to serve outside the agent loop. The MCP exposure serves
    exactly this; the agent merges it with its loop-only providers (delegation,
    consume-side MCP, context). A capability joins by becoming a
    ToolSurfaceFactory.create parameter — deliberately, never implicitly."""

    def __init__(
        self,
        tools_provider: ToolsProvider,
        memory_provider: MemoryProvider,
        skills_provider: SkillsProvider,
    ):
        self.tools_provider = tools_provider
        self.memory_provider = memory_provider
        self.skills_provider = skills_provider

    async def get_tools(self) -> List[Tool]:
        collected = await asyncio.gather(
            self.tools_provider.get_tools(),
            self.memory_provider.get_tools(),
            self.skills_provider.get_tools(),
        )
        return [tool for tools in collected for tool in tools]


class ToolSurfaceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        tools_provider: ToolsProvider,
        memory_provider: MemoryProvider,
        skills_provider: SkillsProvider,
    ) -> ToolSurface:
        return ToolSurface(
            tools_provider=tools_provider,
            memory_provider=memory_provider,
            skills_provider=skills_provider,
        )


AgentDependencyRegistry.register(ToolSurfaceFactory)
