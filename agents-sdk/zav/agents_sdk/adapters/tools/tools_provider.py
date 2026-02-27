from typing import List, Optional, Set

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.tools.tools_source import ToolsSource, ToolsSourceGroup
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool

_PROMPT_SEPARATOR = "\n\n"


class ToolsConfiguration(BaseModel):

    enabled: bool = Field(
        True, description="Enable Python-defined tools from registered sources."
    )
    include_sources: Optional[List[str]] = Field(
        None, description="Allowlist of source names to include."
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Denylist of source names to exclude."
    )
    include_tools: Optional[List[str]] = Field(
        None, description="Allowlist of tool names to include."
    )
    exclude_tools: Optional[List[str]] = Field(
        None, description="Denylist of tool names to exclude."
    )


def _passes_filter(
    name: str,
    include: Optional[Set[str]],
    exclude: Optional[Set[str]],
) -> bool:
    if include is not None and name not in include:
        return False
    if exclude is not None and name in exclude:
        return False
    return True


class ToolsProvider:

    def __init__(
        self,
        sources: List[ToolsSource],
        enabled: bool = True,
        include_sources: Optional[Set[str]] = None,
        exclude_sources: Optional[Set[str]] = None,
        include_tools: Optional[Set[str]] = None,
        exclude_tools: Optional[Set[str]] = None,
    ):
        self.__sources = sources
        self.__enabled = enabled
        self.__include_sources = include_sources
        self.__exclude_sources = exclude_sources
        self.__include_tools = include_tools
        self.__exclude_tools = exclude_tools

    async def get_tools(self) -> List[Tool]:
        if not self.__enabled:
            return []

        tools: List[Tool] = []
        for source in self.__active_sources():
            try:
                source_tools = await source.get_tools()
                for tool in source_tools:
                    if _passes_filter(
                        tool.name, self.__include_tools, self.__exclude_tools
                    ):
                        tools.append(tool)
            except Exception as e:
                logger.error(f"Failed to get tools from {source.source_name}: {e}")
        return tools

    def to_prompt(self) -> str:
        if not self.__enabled:
            return ""
        parts = [
            source.to_prompt()
            for source in self.__active_sources()
            if source.to_prompt()
        ]
        return _PROMPT_SEPARATOR.join(parts)

    def __active_sources(self) -> List[ToolsSource]:
        return [
            source
            for source in self.__sources
            if _passes_filter(
                source.source_name, self.__include_sources, self.__exclude_sources
            )
        ]


class ToolsProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        tools_source_group: ToolsSourceGroup = ToolsSourceGroup(items=[]),
        tools_configuration: ToolsConfiguration = ToolsConfiguration(),
    ) -> ToolsProvider:
        return ToolsProvider(
            sources=tools_source_group.items,
            enabled=tools_configuration.enabled,
            include_sources=(
                set(tools_configuration.include_sources)
                if tools_configuration.include_sources
                else None
            ),
            exclude_sources=(
                set(tools_configuration.exclude_sources)
                if tools_configuration.exclude_sources
                else None
            ),
            include_tools=(
                set(tools_configuration.include_tools)
                if tools_configuration.include_tools
                else None
            ),
            exclude_tools=(
                set(tools_configuration.exclude_tools)
                if tools_configuration.exclude_tools
                else None
            ),
        )
