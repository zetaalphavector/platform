import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from zav.message_bus import Command

from zav.agents_sdk.domain import ChatRequest, RequestHeaders
from zav.agents_sdk.domain.message_inbox import InboxItem


def generate_message_id() -> str:
    return str(uuid.uuid4())


def generate_session_id() -> str:
    return str(uuid.uuid4())


@dataclass
class CreateChatResponse(Command):
    tenant: str
    request_headers: RequestHeaders
    chat_request: ChatRequest
    index_id: Optional[str] = None
    # message_id names this turn; session_id names the conversation. Both are
    # always assigned: a client-supplied session id keeps the conversation's
    # traces (and, when stateful, its persisted state) attached to prior
    # turns, while a freshly minted one names a new conversation the client
    # can echo on its next request.
    message_id: str = field(default_factory=generate_message_id)
    session_id: str = field(default_factory=generate_session_id)
    stateful: bool = False
    resume: bool = False
    # Optional per-turn wall-clock budget in seconds. When set, handle_create
    # runs the turn under it and cancels on expiry. None keeps the unbounded
    # behavior, so existing callers are unaffected; a batch caller such as the
    # pipeline agent processor sets it to bound a slow enhancement.
    request_timeout: Optional[float] = None


@dataclass
class CreateChatStream(CreateChatResponse):
    # What a send to an idle conversation brings. The turn writes it to the
    # inbox only after its agent is found, so a refused caller leaves nothing.
    inbox_items: List[InboxItem] = field(default_factory=list)


@dataclass
class SendChatMessage(Command):
    """Adds ``items`` to the conversation's inbox. When no turn runs, starts
    one with ``chat_request`` (its conversation empty: the turn reads the
    inbox)."""

    tenant: str
    request_headers: RequestHeaders
    session_id: str
    chat_request: ChatRequest
    items: List[InboxItem]
    index_id: Optional[str] = None


@dataclass
class RemoveChatMessage(Command):
    """Takes a sent message back out of the conversation's inbox before a turn
    reads it."""

    tenant: str
    request_headers: RequestHeaders
    session_id: str
    message_id: str


@dataclass
class HandleMCPOAuthCallback(Command):
    state: str
    code: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ListMCPTools(Command):
    tenant: str
    request_headers: RequestHeaders
    mcp_server_identifier: Optional[str]
    index_id: Optional[str] = None
    llm_configuration_name: Optional[str] = None


@dataclass
class CallMCPTool(Command):
    tenant: str
    request_headers: RequestHeaders
    mcp_server_identifier: Optional[str]
    tool_name: str
    arguments: Dict[str, Any]
    index_id: Optional[str] = None
    llm_configuration_name: Optional[str] = None


@dataclass
class StreamRealtimeTranscription(Command):
    tenant: str
    browser_frames: AsyncIterator[Union[str, bytes]]
    requester_uuid: Optional[str] = None
