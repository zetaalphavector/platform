import json
import os
from typing import Optional, cast

from zav.agents_sdk.domain.chat_agent_state_store import (
    ChatAgentStateStore,
    DependencyState,
    construct_agent_state_key,
)


class LocalFileChatAgentStateStore(ChatAgentStateStore):
    def __init__(self, base_path: str):
        self.__base_path = base_path
        os.makedirs(self.__base_path, exist_ok=True)

    def __state_file_path(
        self,
        tenant: str,
        user_uuid: Optional[str],
        session_id: str,
    ) -> str:
        key = construct_agent_state_key(tenant, user_uuid, session_id)
        return os.path.join(self.__base_path, f"{key}.json")

    async def load(
        self,
        tenant: str,
        user_uuid: Optional[str],
        session_id: str,
    ) -> DependencyState:
        file_path = self.__state_file_path(tenant, user_uuid, session_id)
        if not os.path.exists(file_path):
            return {}

        with open(file_path) as state_file:
            state = json.load(state_file)
        if not isinstance(state, dict):
            return {}
        return cast(DependencyState, state)

    async def save(
        self,
        tenant: str,
        user_uuid: Optional[str],
        session_id: str,
        state: DependencyState,
    ) -> None:
        file_path = self.__state_file_path(tenant, user_uuid, session_id)
        # The key carries ``/`` separators, so the per-session subdirectories
        # may not exist yet.
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        temp_file_path = f"{file_path}.tmp"
        with open(temp_file_path, "w") as state_file:
            json.dump(state, state_file)
        os.replace(temp_file_path, file_path)

    async def delete(
        self,
        tenant: str,
        user_uuid: Optional[str],
        session_id: str,
    ) -> None:
        file_path = self.__state_file_path(tenant, user_uuid, session_id)
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
