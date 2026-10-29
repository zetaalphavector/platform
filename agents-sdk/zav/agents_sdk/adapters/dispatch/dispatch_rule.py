from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, ClassVar, Coroutine, Optional

from zav.pydantic_compat import BaseModel

from zav.agents_sdk.domain.agent_dependency import DependencyGroup
from zav.agents_sdk.domain.chat_message import ChatMessage, ConversationContext

EmitFn = Optional[Callable[[str, BaseModel], Coroutine[None, None, None]]]


class DispatchRule(ABC):

    source_name: ClassVar[str]
    # One line for admins choosing sources, e.g. in the agent editor.
    source_description: ClassVar[Optional[str]] = None
    # The tenant gate that hides this source until a tenant enables it.
    source_gate: ClassVar[Optional[str]] = None
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
