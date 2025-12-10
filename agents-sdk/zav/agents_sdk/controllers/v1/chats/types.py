from typing import Any, Dict, List, Optional

from zav.pydantic_compat import PYDANTIC_V2, BaseModel, ConfigDict

from zav.agents_sdk.domain import ChatMessage, ConversationContext, FunctionSpec


class ChatResponseForm(BaseModel):
    agent_identifier: str
    conversation: List[ChatMessage]
    conversation_context: Optional[ConversationContext] = None
    bot_params: Optional[Dict[str, Any]] = None


class ChatResponseItem(ChatResponseForm):
    function_specs: Optional[List[FunctionSpec]] = None

    if PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True)
    else:

        class Config:
            orm_mode = True


class ChatStreamItem(BaseModel):
    event: str
    id: str
    data: ChatMessage
    retry: int
