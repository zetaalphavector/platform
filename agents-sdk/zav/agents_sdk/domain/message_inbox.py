from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from zav.pydantic_compat import BaseModel

from zav.agents_sdk.domain.chat_message import ChatMessage


class ToolDecision(BaseModel):
    """The user's answer to a call waiting for approval."""

    tool_call_id: str
    decision: Literal["allow_once", "allow_session", "deny"]
    # The full edited inputs; replaces the call's params when set.
    params: Optional[Dict[str, Any]] = None


class InboxItem(BaseModel):
    """One entry of a conversation's inbox: a full user message (any content
    parts) or the user's decision on a call waiting for approval."""

    message: Optional[ChatMessage] = None
    tool_decision: Optional[ToolDecision] = None


class MessageInboxStore(ABC):
    """A conversation's inbox, keyed like its agent state. Anyone may append;
    only the turn that holds the conversation's lease takes, so items are read
    once, oldest first, and items no turn read stay for the next one."""

    @abstractmethod
    async def append(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        item: InboxItem,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def take(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> List[InboxItem]:
        """Removes and returns every item, oldest first."""
        raise NotImplementedError

    @abstractmethod
    async def remove(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        message_id: str,
    ) -> bool:
        """Removes the message sent with ``message_id`` before a turn takes
        it. False when it is not there: a turn took it, or none was sent."""
        raise NotImplementedError

    @abstractmethod
    async def has_items(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def clear(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> None:
        raise NotImplementedError
