from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from zav.agents_sdk.adapters.mcp.oauth_repository import (
    MCPOAuthClientInfoRecord,
    MCPOAuthPendingAuthorization,
    MCPOAuthRepository,
    MCPUserIntegrationRecord,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


@dataclass(frozen=True)
class MCPOAuthConnectionState:
    server_name: str
    tokens: Optional[OAuthToken] = None
    pending: Optional[MCPOAuthPendingAuthorization] = None


def _pending_expired(pending: Optional[MCPOAuthPendingAuthorization]) -> bool:
    return bool(pending and pending.expires_at < datetime.now(timezone.utc))


def _is_empty(record: MCPUserIntegrationRecord) -> bool:
    return record.tokens is None and record.pending is None


class MCPOAuthIntegrationStore:
    def __init__(self, repository: MCPOAuthRepository):
        self.__repository = repository

    async def load_user_integrations(
        self, tenant: str, user_uuid: str, server_names: List[str]
    ) -> Dict[str, MCPOAuthConnectionState]:
        records = await self.__repository.list_user_integrations(
            tenant, user_uuid, server_names
        )
        states: Dict[str, MCPOAuthConnectionState] = {}
        for record in records:
            if _pending_expired(record.pending):
                record.pending = None
                if _is_empty(record):
                    await self.__repository.delete_user_integration(
                        record.tenant, record.user_uuid, record.server_name
                    )
                    continue
                await self.__repository.save_user_integration(record)
            if record.tokens or record.pending:
                states[record.server_name] = MCPOAuthConnectionState(
                    server_name=record.server_name,
                    tokens=record.tokens,
                    pending=record.pending,
                )
        return states

    async def load_client_info(
        self, tenant: str, server_names: List[str]
    ) -> Dict[str, OAuthClientInformationFull]:
        records = await self.__repository.list_client_info(tenant, server_names)
        return {record.server_name: record.client_info for record in records}

    async def get_pending_authorization(
        self, state: str
    ) -> Optional[MCPOAuthPendingAuthorization]:
        record = await self.__repository.find_user_integration_by_pending_state(state)
        if record is None:
            return None
        if _pending_expired(record.pending):
            record.pending = None
            if _is_empty(record):
                await self.__repository.delete_user_integration(
                    record.tenant, record.user_uuid, record.server_name
                )
            else:
                await self.__repository.save_user_integration(record)
            return None
        return record.pending

    async def save_pending_authorization(
        self,
        tenant: str,
        user_uuid: str,
        server_name: str,
        pending: MCPOAuthPendingAuthorization,
    ) -> None:
        record = await self.__repository.get_user_integration(
            tenant, user_uuid, server_name
        ) or MCPUserIntegrationRecord(
            tenant=tenant, user_uuid=user_uuid, server_name=server_name
        )
        record.pending = pending
        await self.__repository.save_user_integration(record)

    async def save_tokens(
        self, tenant: str, user_uuid: str, server_name: str, tokens: OAuthToken
    ) -> None:
        record = await self.__repository.get_user_integration(
            tenant, user_uuid, server_name
        ) or MCPUserIntegrationRecord(
            tenant=tenant, user_uuid=user_uuid, server_name=server_name
        )
        record.tokens = tokens
        record.pending = None
        await self.__repository.save_user_integration(record)

    async def set_client_info(
        self, tenant: str, server_name: str, client_info: OAuthClientInformationFull
    ) -> None:
        await self.__repository.save_client_info(
            MCPOAuthClientInfoRecord(
                tenant=tenant, server_name=server_name, client_info=client_info
            )
        )

    async def clear_pending_authorization(self, state: str) -> None:
        record = await self.__repository.find_user_integration_by_pending_state(state)
        if record is None:
            return
        record.pending = None
        if _is_empty(record):
            await self.__repository.delete_user_integration(
                record.tenant, record.user_uuid, record.server_name
            )
            return
        await self.__repository.save_user_integration(record)

    async def clear_tokens(self, tenant: str, user_uuid: str, server_name: str) -> None:
        record = await self.__repository.get_user_integration(
            tenant, user_uuid, server_name
        )
        if record is None:
            return
        record.tokens = None
        if _is_empty(record):
            await self.__repository.delete_user_integration(
                tenant, user_uuid, server_name
            )
            return
        await self.__repository.save_user_integration(record)

    async def delete_user_integration(
        self, tenant: str, user_uuid: str, server_name: str
    ) -> bool:
        return await self.__repository.delete_user_integration(
            tenant, user_uuid, server_name
        )


class MCPOAuthIntegrationStoreFactory(AgentDependencyFactory):
    def __init__(self, mcp_oauth_store: MCPOAuthIntegrationStore):
        self.__mcp_oauth_store = mcp_oauth_store

    def create(self) -> MCPOAuthIntegrationStore:  # type: ignore[override]
        return self.__mcp_oauth_store


class PersistentTokenStorage(TokenStorage):

    def __init__(
        self,
        store: MCPOAuthIntegrationStore,
        tenant: str,
        user_uuid: str,
        server_name: str,
        tokens: Optional[OAuthToken] = None,
        client_info: Optional[OAuthClientInformationFull] = None,
    ):
        self.__store = store
        self.__tenant = tenant
        self.__server_name = server_name
        self.__user_uuid = user_uuid
        self.__tokens = tokens
        self.__client_info = client_info

    async def get_tokens(self) -> Optional[OAuthToken]:
        return self.__tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.__tokens = tokens
        await self.__store.save_tokens(
            self.__tenant, self.__user_uuid, self.__server_name, tokens
        )

    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        return self.__client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.__client_info = client_info
        await self.__store.set_client_info(
            self.__tenant, self.__server_name, client_info
        )
