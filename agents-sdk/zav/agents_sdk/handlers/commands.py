import uuid
from dataclasses import dataclass, field
from typing import Optional

from zav.message_bus import Command

from zav.agents_sdk.domain import ChatRequest, RequestHeaders


def generate_message_id() -> str:
    return str(uuid.uuid4())


@dataclass
class CreateChatResponse(Command):
    tenant: str
    request_headers: RequestHeaders
    chat_request: ChatRequest
    index_id: Optional[str] = None
    message_id: str = field(default_factory=generate_message_id)


@dataclass
class CreateChatStream(CreateChatResponse):
    pass


@dataclass
class CreateBufferedChatStream(CreateChatStream):
    pass


@dataclass
class GetChatStreamBuffer(Command):
    message_id: str
    tenant: str
    requester_uuid: Optional[str] = None


@dataclass
class CancelChatStream(Command):
    message_id: str
    tenant: str
    requester_uuid: Optional[str] = None
