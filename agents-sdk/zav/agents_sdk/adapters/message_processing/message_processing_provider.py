from typing import List

from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import ChatResponse
from zav.agents_sdk.adapters.message_processing.message_processor import (
    MessageProcessor,
    MessageProcessorGroup,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage


class MessageProcessingProvider:

    def __init__(self, processors: List[MessageProcessor]):
        self.__processors = processors

    async def process(
        self, response: ChatResponse, message: ChatMessage
    ) -> ChatMessage:
        for processor in self.__processors:
            message = await processor.process(response, message)
        return message


class MessageProcessingProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        message_processor_group: MessageProcessorGroup = MessageProcessorGroup(
            items=[]
        ),
    ) -> MessageProcessingProvider:
        return MessageProcessingProvider(
            processors=message_processor_group.items,
        )
