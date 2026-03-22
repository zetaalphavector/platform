from abc import ABC, abstractmethod
from typing import ClassVar

from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import ChatResponse
from zav.agents_sdk.domain.agent_dependency import DependencyGroup
from zav.agents_sdk.domain.chat_message import ChatMessage


class MessageProcessor(ABC):
    """Transforms outgoing messages before they are yielded to the caller.

    Extend this to add post-processing to LLM responses — e.g. citation
    resolution, evidence extraction, content rewriting.

    Receives both the raw ``ChatResponse`` from the LLM client and the
    already-converted ``ChatMessage``.  The response carries metadata
    (completion sender, function_call_request) that processors may need
    to decide whether to act.
    """

    source_name: ClassVar[str]

    @abstractmethod
    async def process(
        self, response: ChatResponse, message: ChatMessage
    ) -> ChatMessage:
        raise NotImplementedError


class MessageProcessorGroup(DependencyGroup[MessageProcessor]):
    __collects__ = MessageProcessor
