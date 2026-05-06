from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import is_source_active
from zav.agents_sdk.adapters.message_processing.message_processor import (
    MessageProcessor,
    MessageProcessorGroup,
    StreamItem,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage


class MessageProcessingProviderConfiguration(BaseModel):
    enabled: bool = Field(True, description="Enable message post-processing.")
    include_sources: Optional[List[str]] = Field(
        None, description="Allowlist of processor source names to include."
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Denylist of processor source names to exclude."
    )


class MessageProcessingProvider:

    def __init__(
        self,
        processors: List[MessageProcessor],
        enabled: bool,
        include_sources: Optional[Set[str]] = None,
        exclude_sources: Optional[Set[str]] = None,
    ):
        self.__processors = processors
        self.__enabled = enabled
        self.__include_sources = include_sources
        self.__exclude_sources = exclude_sources
        self.__cached_active: Optional[List[MessageProcessor]] = None

    def describe_loaded(self) -> Dict[str, Any]:
        return {
            "enabled": self.__enabled,
            "sources": [p.source_name for p in self.__active_processors()],
        }

    async def process_stream(
        self, stream: AsyncGenerator[StreamItem, None]
    ) -> AsyncGenerator[ChatMessage, None]:
        if not self.__enabled:
            async for _, message in stream:
                yield message
            return

        processors = self.__active_processors()
        if not processors:
            async for _, message in stream:
                yield message
            return

        current: AsyncGenerator[StreamItem, None] = stream
        for processor in processors:
            current = processor.process_stream(current)

        async for _, message in current:
            yield message

    def __active_processors(self) -> List[MessageProcessor]:
        if self.__cached_active is not None:
            return self.__cached_active
        self.__cached_active = [
            p
            for p in self.__processors
            if is_source_active(
                p.source_name,
                p.enabled,
                self.__include_sources,
                self.__exclude_sources,
            )
        ]
        return self.__cached_active


class MessageProcessingProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        message_processor_group: MessageProcessorGroup = MessageProcessorGroup(
            items=[]
        ),
        message_processing_provider_configuration: MessageProcessingProviderConfiguration = MessageProcessingProviderConfiguration(),  # noqa: E501
    ) -> MessageProcessingProvider:
        return MessageProcessingProvider(
            processors=message_processor_group.items,
            enabled=message_processing_provider_configuration.enabled,
            include_sources=(
                set(message_processing_provider_configuration.include_sources)
                if message_processing_provider_configuration.include_sources
                else None
            ),
            exclude_sources=(
                set(message_processing_provider_configuration.exclude_sources)
                if message_processing_provider_configuration.exclude_sources
                else None
            ),
        )
