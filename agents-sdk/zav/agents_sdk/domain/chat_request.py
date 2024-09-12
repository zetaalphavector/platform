from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from zav.agents_sdk.domain.chat_message import ChatMessage, ConversationContext


# TODO: How can the client define an agent_identifier for subagents so we can use
# the same subagent implementation but with different configs?
class ChatRequest(BaseModel):
    agent_identifier: str
    conversation: List[ChatMessage]
    conversation_context: Optional[ConversationContext] = None
    bot_params: Optional[Dict[str, Any]] = None


class ChatStreamRequest(BaseModel):
    agent_identifier: str
    conversation: List[ChatMessage]
    conversation_context: Optional[ConversationContext] = None
    bot_params: Optional[Dict[str, Any]] = None
