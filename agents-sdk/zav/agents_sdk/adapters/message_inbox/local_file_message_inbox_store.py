import fcntl
import hashlib
import json
import os
from typing import List, Optional

from zav.pydantic_compat import PYDANTIC_V2

from zav.agents_sdk.domain.message_inbox import InboxItem, MessageInboxStore


class LocalFileMessageInboxStore(MessageInboxStore):
    """File-backed inbox for local/dev: one JSON-lines file per conversation.
    Appends and takes run under an exclusive ``flock`` on the conversation's
    lock file, so processes sharing the directory never lose or re-read an
    item."""

    def __init__(self, base_path: str):
        self.__base_path = base_path
        os.makedirs(self.__base_path, exist_ok=True)

    def __inbox_file_path(
        self, tenant: str, requester_uuid: Optional[str], session_id: str
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
        return os.path.join(self.__base_path, f"{file_name}.jsonl")

    async def append(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        item: InboxItem,
    ) -> None:
        file_path = self.__inbox_file_path(tenant, requester_uuid, session_id)
        line = item.model_dump_json() if PYDANTIC_V2 else item.json()
        with open(f"{file_path}.lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            with open(file_path, "a") as inbox_file:
                inbox_file.write(f"{line}\n")

    async def take(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> List[InboxItem]:
        file_path = self.__inbox_file_path(tenant, requester_uuid, session_id)
        if not os.path.exists(file_path):
            return []
        with open(f"{file_path}.lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            with open(file_path) as inbox_file:
                lines = [line for line in inbox_file.read().splitlines() if line]
            os.remove(file_path)
        if PYDANTIC_V2:
            return [InboxItem.model_validate_json(line) for line in lines]
        return [InboxItem.parse_raw(line) for line in lines]

    async def remove(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        message_id: str,
    ) -> bool:
        file_path = self.__inbox_file_path(tenant, requester_uuid, session_id)
        with open(f"{file_path}.lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                with open(file_path) as inbox_file:
                    lines = [line for line in inbox_file.read().splitlines() if line]
            except FileNotFoundError:
                return False
            kept = [line for line in lines if self.__message_id(line) != message_id]
            if len(kept) == len(lines):
                return False
            with open(file_path, "w") as inbox_file:
                inbox_file.writelines(f"{line}\n" for line in kept)
        return True

    async def has_items(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> bool:
        file_path = self.__inbox_file_path(tenant, requester_uuid, session_id)
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    async def clear(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> None:
        file_path = self.__inbox_file_path(tenant, requester_uuid, session_id)
        with open(f"{file_path}.lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass

    @staticmethod
    def __message_id(line: str) -> Optional[str]:
        item = (
            InboxItem.model_validate_json(line)
            if PYDANTIC_V2
            else InboxItem.parse_raw(line)
        )
        return item.message.message_id if item.message is not None else None
