from typing import Optional

from zav.api import AppResource

from zav.agents_sdk.domain.chat_conversation_store import ChatConversationStore

get_chat_conversation_store = AppResource[Optional[ChatConversationStore]](
    "chat_conversation_store"
)
