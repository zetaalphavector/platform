from zav.llm_domain import (
    ToolSearchCatalogue,
    ToolSearchMethodSettings,
    ToolSearchMode,
    ToolSearchOptions,
)
from zav.pydantic_compat import Field

from zav.agents_sdk.adapters.tool_search.tool_search_source import ToolSearchSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class ClientToolSearchSourceConfiguration(ToolSearchOptions):
    enabled: bool = Field(
        False,
        description="Run our tool search through the vendor's tool search protocol.",
    )


class ClientToolSearchSource(ToolSearchSource):
    source_name = ToolSearchMode.CLIENT.value
    source_description = (
        "Our tool search in the vendor's protocol (OpenAI Responses, Anthropic): "
        "found tools load without changing the tools sent up front."
    )

    def __init__(
        self, enabled: bool, catalogue: ToolSearchCatalogue, max_results: int
    ) -> None:
        self.enabled = enabled
        self.__catalogue = catalogue
        self.__max_results = max_results

    def settings(self) -> ToolSearchMethodSettings:
        return ToolSearchMethodSettings(
            method=ToolSearchMode.CLIENT,
            catalogue=self.__catalogue,
            max_results=self.__max_results,
        )


class ClientToolSearchSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        client_tool_search_source_configuration: (
            ClientToolSearchSourceConfiguration
        ) = ClientToolSearchSourceConfiguration(),
    ) -> ClientToolSearchSource:
        configuration = client_tool_search_source_configuration
        return ClientToolSearchSource(
            enabled=configuration.enabled,
            catalogue=configuration.catalogue,
            max_results=configuration.max_results,
        )
