import hashlib
import json
import os
import uuid
from typing import Optional

from zav.pydantic_compat import PYDANTIC_V2

from zav.agents_sdk.domain.chat_agent import ChatMessage
from zav.agents_sdk.domain.chat_stream_recording_store import (
    ChatStreamRecordingStore,
    RecordedStreamEvent,
)


class LocalFileChatStreamRecordingStore(ChatStreamRecordingStore):
    """File-backed recording store for local/dev and the perf harness.

    Writes one fully-overwritten JSON file per conversation — the dev twin of
    the production object-store blob, which holds the latest cumulative
    snapshot of the conversation's current turn rather than an append log.
    Every perf server points at the same directory, so a turn recorded by one
    server is readable by another: the cross-pod stand-in for the durable
    stream record production will eventually use.

    Each ``put`` overwrites the record atomically via a temp file +
    ``os.replace`` so a reconnecting pod sharing the directory never sees a torn
    read.
    """

    def __init__(self, base_path: str):
        self.__base_path = base_path
        os.makedirs(self.__base_path, exist_ok=True)

    def __recording_file_path(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> str:
        key = json.dumps(
            {
                "session_id": session_id,
                "tenant": tenant,
                "requester_uuid": requester_uuid,
            },
            sort_keys=True,
        )
        file_name = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.__base_path, f"{file_name}.json")

    async def put(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        event: RecordedStreamEvent,
    ) -> None:
        # Reject a write from a producer fenced by a newer owner (epoch below the
        # current record's). Best-effort here — this file store is the local/dev
        # twin; production fencing is the SQL store's same-row compare-and-swap.
        current = await self.read(tenant, requester_uuid, session_id)
        if current is not None and event.epoch < current.epoch:
            return
        file_path = self.__recording_file_path(tenant, requester_uuid, session_id)
        if PYDANTIC_V2:
            message_json = event.message.model_dump(mode="json")
        else:
            message_json = json.loads(event.message.json())
        payload = json.dumps(
            {
                "index": event.index,
                "message": message_json,
                "message_id": event.message_id,
                "epoch": event.epoch,
            }
        )
        tmp_path = f"{file_path}.{uuid.uuid4().hex}.tmp"
        with open(tmp_path, "w") as recording_file:
            recording_file.write(payload)
        os.replace(tmp_path, file_path)

    async def read(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> Optional[RecordedStreamEvent]:
        file_path = self.__recording_file_path(tenant, requester_uuid, session_id)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path) as recording_file:
                record = json.loads(recording_file.read())
        except (OSError, json.JSONDecodeError):
            return None
        if PYDANTIC_V2:
            message = ChatMessage.model_validate(record["message"])
        else:
            message = ChatMessage.parse_obj(record["message"])
        return RecordedStreamEvent(
            index=record["index"],
            message=message,
            message_id=record["message_id"],
            epoch=record.get("epoch", 0),
        )

    async def delete(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        message_id: str,
    ) -> None:
        current = await self.read(tenant, requester_uuid, session_id)
        if current is None or current.message_id != message_id:
            return
        file_path = self.__recording_file_path(tenant, requester_uuid, session_id)
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    def clear(self) -> None:
        if not os.path.exists(self.__base_path):
            return
        for file_name in os.listdir(self.__base_path):
            if file_name.endswith(".json") or file_name.endswith(".tmp"):
                os.remove(os.path.join(self.__base_path, file_name))
