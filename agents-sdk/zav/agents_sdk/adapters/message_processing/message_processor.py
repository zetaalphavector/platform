from abc import ABC, abstractmethod
from typing import AsyncGenerator, ClassVar, Tuple

from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import ChatResponse
from zav.agents_sdk.domain.agent_dependency import DependencyGroup
from zav.agents_sdk.domain.chat_message import ChatMessage

StreamItem = Tuple[ChatResponse, ChatMessage]


class MessageProcessor(ABC):
    """Transforms an outgoing message stream before it is yielded to the caller.

    Receives the full async generator of (ChatResponse, ChatMessage) pairs
    and yields transformed (ChatResponse, ChatMessage) pairs. This gives
    each processor full control over buffering, batching, or waiting for
    the stream to complete before acting.
    """

    source_name: ClassVar[str]
    enabled: bool

    @abstractmethod
    async def process_stream(
        self, stream: AsyncGenerator[StreamItem, None]
    ) -> AsyncGenerator[StreamItem, None]:
        raise NotImplementedError
        yield  # pragma: no cover


class MessageProcessorGroup(DependencyGroup[MessageProcessor]):
    __collects__ = MessageProcessor
