import uuid
from dataclasses import dataclass, field
from typing import Optional

from zav.message_bus import Command

from zav.agents_sdk.domain import ChatRequest, RequestHeaders


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


@dataclass
class CreateChatStream(CreateChatResponse):
    pass


@dataclass
class HandleMCPOAuthCallback(Command):
    state: str
    code: Optional[str] = None
    error: Optional[str] = None
