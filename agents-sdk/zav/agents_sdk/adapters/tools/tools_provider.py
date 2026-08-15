import inspect
from typing import Any, Dict, List, Optional, Set

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import is_source_active, passes_name_filter
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource, ToolsSourceGroup
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool

_PROMPT_SEPARATOR = "\n\n"


class ToolsProviderConfiguration(BaseModel):

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
        self.__resolved: Optional[Dict[str, List[Tool]]] = None
        self.__cached_active: Optional[List[ToolsSource]] = None
        self.__cleaned_up = False

    async def get_tools(self) -> List[Tool]:
        if not self.__enabled:
            return []
        resolved = await self.__resolve_source_tools()
        return [t for tools in resolved.values() for t in tools]

    async def __resolve_source_tools(
        self,
    ) -> Dict[str, List[Tool]]:
        if self.__resolved is not None:
            return self.__resolved
        self.__resolved = {}
        for source in self.__active_sources():
            try:
                source_tools = await source.get_tools()
                self.__resolved[source.source_name] = [
                    t
                    for t in source_tools
                    if passes_name_filter(
                        t.name,
                        self.__include_tools,
                        self.__exclude_tools,
                    )
                ]
            except Exception as e:
                logger.error(f"Failed to get tools from {source.source_name}: {e}")
                self.__resolved[source.source_name] = []
        return self.__resolved

    async def to_prompt(self) -> str:
        if not self.__enabled:
            return ""
        parts = []
        for source in self.__active_sources():
            prompt = await source.to_prompt()
            if prompt:
                parts.append(prompt)
        return _PROMPT_SEPARATOR.join(parts)

    async def describe_loaded(self) -> Dict[str, Any]:
        resolved = await self.__resolve_source_tools()
        return {
            "enabled": self.__enabled,
            "sources": list(resolved.keys()),
            "tools_by_source": {
                name: [t.name for t in tools] for name, tools in resolved.items()
            },
        }

    async def cleanup(self) -> None:
        if self.__cleaned_up:
            return

        self.__cleaned_up = True
        for source in self.__sources:
            aclose = getattr(source, "aclose", None)
            if not aclose or not callable(aclose):
                continue
            try:
                result = aclose()
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                logger.warning(
                    f"Error cleaning up tools source {source.source_name}: {e}"
                )

        self.__resolved = None
        self.__cached_active = None

    def __active_sources(self) -> List[ToolsSource]:
        if self.__cached_active is not None:
            return self.__cached_active
        self.__cached_active = [
            source
            for source in self.__sources
            if is_source_active(
                source.source_name,
                source.enabled,
                self.__include_sources,
                self.__exclude_sources,
            )
        ]
        return self.__cached_active


class ToolsProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        tools_source_group: ToolsSourceGroup = ToolsSourceGroup(items=[]),
        tools_provider_configuration: ToolsProviderConfiguration = (
            ToolsProviderConfiguration()
        ),
    ) -> ToolsProvider:
        return ToolsProvider(
            sources=tools_source_group.items,
            enabled=tools_provider_configuration.enabled,
            include_sources=(
                set(tools_provider_configuration.include_sources)
                if tools_provider_configuration.include_sources
                else None
            ),
            exclude_sources=(
                set(tools_provider_configuration.exclude_sources)
                if tools_provider_configuration.exclude_sources
                else None
            ),
            include_tools=(
                set(tools_provider_configuration.include_tools)
                if tools_provider_configuration.include_tools
                else None
            ),
            exclude_tools=(
                set(tools_provider_configuration.exclude_tools)
                if tools_provider_configuration.exclude_tools
                else None
            ),
        )
