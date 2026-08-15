from typing import Optional

from zav.api import AppResource

from zav.agents_sdk.domain.chat_stream_supervision_store import (
    ChatStreamSupervisionStore,
)

get_chat_stream_supervision_store = AppResource[Optional[ChatStreamSupervisionStore]](
    "chat_stream_supervision_store"
)
