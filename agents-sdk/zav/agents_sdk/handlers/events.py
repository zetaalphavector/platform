from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from zav.message_bus import Event

from zav.agents_sdk.domain.chat_message import ChatMessage, ConversationContext


@dataclass
class EventBase(Event):
    tenant: str
    index_id: Optional[str]
    request_headers: Dict[str, Any]


@dataclass
class CreatedAgentRequest(EventBase):
    agent_identifier: str
    conversation: List[ChatMessage]
    conversation_context: Optional[ConversationContext]
    bot_params: Optional[Dict[str, Any]]


@dataclass
class ConversationUpdated(EventBase):
    # Fired when a stateful turn completes. Carries ids only (no transcript) so
    # the Pulsar message stays small; the consumer re-reads the conversation
    # from the DB, markdownifies it, and upserts the saved-results note. The
    # requester is carried in request_headers (inherited from EventBase).
    session_id: str
    agent_identifier: str
