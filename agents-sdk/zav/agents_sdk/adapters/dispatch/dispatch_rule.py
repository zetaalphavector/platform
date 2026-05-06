from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, ClassVar, Coroutine, Optional

from zav.pydantic_compat import BaseModel

from zav.agents_sdk.domain.agent_dependency import DependencyGroup
from zav.agents_sdk.domain.chat_message import ChatMessage, ConversationContext

EmitFn = Optional[Callable[[str, BaseModel], Coroutine[None, None, None]]]


class DispatchRule(ABC):

    source_name: ClassVar[str]
    enabled: bool

    @abstractmethod
    def matches(
        self,
        conversation_context: Optional[ConversationContext],
    ) -> bool: ...

    @abstractmethod
    async def execute(
        self,
        conversation: list[ChatMessage],
        conversation_context: Optional[ConversationContext],
        emit: EmitFn = None,
    ) -> AsyncGenerator[ChatMessage, None]: ...


class DispatchRuleGroup(DependencyGroup[DispatchRule]):
    __collects__ = DispatchRule
