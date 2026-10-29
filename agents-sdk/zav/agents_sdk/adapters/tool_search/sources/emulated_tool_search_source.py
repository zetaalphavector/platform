from zav.llm_domain import (
    ToolSearchCatalogue,
    ToolSearchMethodSettings,
    ToolSearchMode,
    ToolSearchOptions,
)
from zav.pydantic_compat import Field

from zav.agents_sdk.adapters.tool_search.tool_search_source import ToolSearchSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class EmulatedToolSearchSourceConfiguration(ToolSearchOptions):
    enabled: bool = Field(
        False, description="Give the model a search tool to load tools with."
    )


class EmulatedToolSearchSource(ToolSearchSource):
    source_name = ToolSearchMode.EMULATED.value
    source_description = (
        "A search tool any model can call; found tools are added to the next request."
    )

    def __init__(
        self, enabled: bool, catalogue: ToolSearchCatalogue, max_results: int
    ) -> None:
        self.enabled = enabled
        self.__catalogue = catalogue
        self.__max_results = max_results

    def settings(self) -> ToolSearchMethodSettings:
        return ToolSearchMethodSettings(
            method=ToolSearchMode.EMULATED,
            catalogue=self.__catalogue,
            max_results=self.__max_results,
        )


class EmulatedToolSearchSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        emulated_tool_search_source_configuration: (
            EmulatedToolSearchSourceConfiguration
        ) = EmulatedToolSearchSourceConfiguration(),
    ) -> EmulatedToolSearchSource:
        configuration = emulated_tool_search_source_configuration
        return EmulatedToolSearchSource(
            enabled=configuration.enabled,
            catalogue=configuration.catalogue,
            max_results=configuration.max_results,
        )
