from abc import ABC, abstractmethod
from typing import List

from zav.agents_sdk.adapters.llm_models.chat_completion_types import ChatCompletion
from zav.agents_sdk.domain.chat_message import ChatMessage


class QueuedMessages(ABC):
    """The messages the user sends while a turn runs, as the root agent's turn
    reads them before each model call. ``QueuedMessagesProvider`` implements
    it; the runner sees only this because the provider's package registers
    sources that import the context provider, which imports the chat
    completion client."""

    @property
    @abstractmethod
    def enabled(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def take(self) -> List[ChatMessage]:
        raise NotImplementedError

    @abstractmethod
    async def has_messages(self) -> bool:
        """Whether a message waits for the model; it stays for ``take``."""
        raise NotImplementedError

    @abstractmethod
    async def frame(self, messages: List[ChatMessage]) -> List[ChatCompletion]:
        raise NotImplementedError
