from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional

from zav.agents_sdk.adapters.llm_models.chat_completion_types import ChatCompletion
from zav.agents_sdk.domain.agent_dependency import DependencyGroup
from zav.agents_sdk.domain.chat_message import ChatMessage


class QueuedMessagesSource(ABC):
    """Decides how the messages the user sends while a turn runs reach the
    model: it turns them into the completions added to the transcript before
    the next model call."""

    source_name: ClassVar[str]
    # One line for admins choosing sources, e.g. in the agent editor.
    source_description: ClassVar[Optional[str]] = None
    # The tenant gate that hides this source until a tenant enables it.
    source_gate: ClassVar[Optional[str]] = None
    enabled: bool

    @abstractmethod
    async def frame(self, messages: List[ChatMessage]) -> List[ChatCompletion]:
        raise NotImplementedError


class QueuedMessagesSourceGroup(DependencyGroup[QueuedMessagesSource]):
    __collects__ = QueuedMessagesSource
