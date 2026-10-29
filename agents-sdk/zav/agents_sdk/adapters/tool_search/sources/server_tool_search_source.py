from zav.llm_domain import ToolSearchMethodSettings, ToolSearchMode
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.tool_search.tool_search_source import ToolSearchSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class ServerToolSearchSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Let the vendor search the tools.")


class ServerToolSearchSource(ToolSearchSource):
    source_name = ToolSearchMode.SERVER.value
    source_description = (
        "The vendor's built-in tool search (OpenAI Responses, Anthropic): the "
        "vendor lists the tools and searches them."
    )

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def settings(self) -> ToolSearchMethodSettings:
        return ToolSearchMethodSettings(method=ToolSearchMode.SERVER)


class ServerToolSearchSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        server_tool_search_source_configuration: (
            ServerToolSearchSourceConfiguration
        ) = ServerToolSearchSourceConfiguration(),
    ) -> ServerToolSearchSource:
        return ServerToolSearchSource(
            enabled=server_tool_search_source_configuration.enabled
        )
