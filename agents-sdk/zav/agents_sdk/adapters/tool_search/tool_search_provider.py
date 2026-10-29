from typing import List, Optional, Set

from zav.llm_domain import ToolSearchMethodSettings
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import active_sources
from zav.agents_sdk.adapters.tool_search.tool_search_source import (
    ToolSearchSource,
    ToolSearchSourceGroup,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class ToolSearchProviderConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description="Load on-demand tools through tool search. Off sends every "
        "tool up front.",
    )
    include_sources: Optional[List[str]] = Field(
        None,
        description="The tool search methods to use, preferred first: `server` "
        "(the vendor searches), `client` (the vendor's protocol, our search), "
        "`emulated` (a search tool any model can call). Unset: every enabled "
        "method in that order.",
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Tool search methods never to use."
    )


class ToolSearchProvider:
    """The agent's tool search methods, preferred first. Which tools load on
    demand is the tool policy's answer; this only says how they are found."""

    def __init__(
        self,
        sources: List[ToolSearchSource],
        enabled: bool,
        include_sources: Optional[List[str]],
        exclude_sources: Optional[Set[str]],
    ) -> None:
        active = active_sources(
            sources,
            enabled,
            set(include_sources) if include_sources is not None else None,
            exclude_sources,
        )
        order = include_sources or [source.source_name for source in sources]
        self.__sources = sorted(active, key=lambda s: order.index(s.source_name))

    def methods(self) -> List[ToolSearchMethodSettings]:
        return [source.settings() for source in self.__sources]


class ToolSearchProviderFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        tool_search_source_group: ToolSearchSourceGroup = ToolSearchSourceGroup(
            items=[]
        ),
        tool_search_provider_configuration: ToolSearchProviderConfiguration = (
            ToolSearchProviderConfiguration()
        ),
    ) -> ToolSearchProvider:
        configuration = tool_search_provider_configuration
        return ToolSearchProvider(
            sources=tool_search_source_group.items,
            enabled=configuration.enabled,
            include_sources=configuration.include_sources,
            exclude_sources=(
                set(configuration.exclude_sources)
                if configuration.exclude_sources
                else None
            ),
        )
