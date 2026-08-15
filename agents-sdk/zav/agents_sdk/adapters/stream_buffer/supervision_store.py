import fcntl
import hashlib
import json
import os
import uuid
from typing import Optional

from zav.agents_sdk.domain.chat_stream_supervision_store import (
    ChatStreamStatus,
    ChatStreamSupervision,
    ChatStreamSupervisionStore,
)


class LocalFileChatStreamSupervisionStore(ChatStreamSupervisionStore):
    """File-backed supervision store for local/dev and the perf harness.

    Writes one fully-overwritten JSON file per conversation, rewritten on every
    heartbeat, so each write lands atomically via a temp file + ``os.replace``
    to avoid torn reads by a reconnecting pod sharing the same directory.

    ``compare_and_swap`` runs its read-compare-write under an exclusive
    ``flock`` on a per-conversation lock file, so two processes sharing the
    directory cannot both clear the precondition and clobber each other's
    claim. That gives the file backend the same atomic compare-and-swap the
    production backend gets from a serializable transaction — which is what
    lets one pod fence another's stale lease when it takes a turn over.
    """

    def __init__(self, base_path: str):
        self.__base_path = base_path
        os.makedirs(self.__base_path, exist_ok=True)

    def __supervision_file_path(
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

    async def compare_and_swap(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        expected_epoch: Optional[int],
        supervision: ChatStreamSupervision,
    ) -> bool:
        file_path = self.__supervision_file_path(tenant, requester_uuid, session_id)
        # Serialize the read-compare-write so two processes sharing the
        # directory cannot both clear the precondition and clobber each other's
        # claim — the file-backed equivalent of the production backend's
        # serializable compare-and-swap.
        with open(f"{file_path}.lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            current = await self.read(tenant, requester_uuid, session_id)
            current_epoch = current.epoch if current is not None else None
            if current_epoch != expected_epoch:
                return False
            payload = json.dumps(
                {
                    "status": supervision.status.value,
                    "heartbeat_at": supervision.heartbeat_at,
                    "message_id": supervision.message_id,
                    "epoch": supervision.epoch,
                }
            )
            tmp_path = f"{file_path}.{uuid.uuid4().hex}.tmp"
            with open(tmp_path, "w") as supervision_file:
                supervision_file.write(payload)
            os.replace(tmp_path, file_path)
            return True

    async def read(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> Optional[ChatStreamSupervision]:
        file_path = self.__supervision_file_path(tenant, requester_uuid, session_id)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path) as supervision_file:
                record = json.loads(supervision_file.read())
        except (OSError, json.JSONDecodeError):
            return None
        return ChatStreamSupervision(
            status=ChatStreamStatus(record["status"]),
            heartbeat_at=record["heartbeat_at"],
            message_id=record["message_id"],
            epoch=record.get("epoch", 0),
        )

    async def delete(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        message_id: str,
        epoch: int,
    ) -> None:
        file_path = self.__supervision_file_path(tenant, requester_uuid, session_id)
        # Same lock as compare_and_swap: the occupancy check and the removal
        # must be one atomic step, or a claim landing in between would be
        # deleted by a reaper for an already-superseded turn.
        with open(f"{file_path}.lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            current = await self.read(tenant, requester_uuid, session_id)
            if (
                current is None
                or current.message_id != message_id
                or current.epoch != epoch
            ):
                return
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass

    def clear(self) -> None:
        if not os.path.exists(self.__base_path):
            return
        for file_name in os.listdir(self.__base_path):
            if file_name.endswith(".json") or file_name.endswith(".lock"):
                os.remove(os.path.join(self.__base_path, file_name))
