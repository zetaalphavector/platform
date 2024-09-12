from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from zav.agents_sdk.domain import ChatMessage, ConversationContext, FunctionSpec


class ChatResponseForm(BaseModel):
    agent_identifier: str
    conversation: List[ChatMessage]
    conversation_context: Optional[ConversationContext] = None
    bot_params: Optional[Dict[str, Any]] = None


class ChatResponseItem(ChatResponseForm):
    function_specs: Optional[List[FunctionSpec]] = None

    class Config:
        orm_mode = True


class ChatStreamItem(BaseModel):
    event: str
    id: str
    data: ChatMessage
    retry: int
