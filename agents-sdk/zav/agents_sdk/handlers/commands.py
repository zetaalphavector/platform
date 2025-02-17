from dataclasses import dataclass
from typing import Optional

from zav.message_bus import Command

from zav.agents_sdk.domain import ChatRequest, RequestHeaders


@dataclass
class CreateChatResponse(Command):
    tenant: str
    request_headers: RequestHeaders
    chat_request: ChatRequest
    index_id: Optional[str] = None


@dataclass
class CreateChatStream(CreateChatResponse):
    pass
