from typing import Optional

from zav.api import AppResource

from zav.agents_sdk.domain.message_inbox import MessageInboxStore

get_message_inbox_store = AppResource[Optional[MessageInboxStore]](
    "message_inbox_store"
)
