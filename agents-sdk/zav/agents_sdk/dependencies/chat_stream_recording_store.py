from typing import Optional

from zav.api import AppResource

from zav.agents_sdk.domain.chat_stream_recording_store import ChatStreamRecordingStore

get_chat_stream_recording_store = AppResource[Optional[ChatStreamRecordingStore]](
    "chat_stream_recording_store"
)
