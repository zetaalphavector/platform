from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.system_prompt.system_prompt_source import (
    SystemPromptSource,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class InMemorySystemPromptSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable the in-memory system prompt.")
    text: str = Field("", description="The system prompt text.")


class InMemorySystemPromptSource(SystemPromptSource):

    source_name = "in_memory_system_prompt"
    source_description = "System prompt text written in this configuration."

    def __init__(self, enabled: bool, text: str):
        self.enabled = enabled
        self.__text = text

    async def to_prompt(self) -> str:
        return self.__text.strip()


class InMemorySystemPromptSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        in_memory_system_prompt_source_configuration: InMemorySystemPromptSourceConfiguration = InMemorySystemPromptSourceConfiguration(),  # noqa: E501
    ) -> InMemorySystemPromptSource:
        return InMemorySystemPromptSource(
            enabled=in_memory_system_prompt_source_configuration.enabled,
            text=in_memory_system_prompt_source_configuration.text,
        )
