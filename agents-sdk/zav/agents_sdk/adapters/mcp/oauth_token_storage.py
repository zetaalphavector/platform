from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from zav.agents_sdk.adapters.mcp.oauth_repository import (
    MCPOAuthClientInfoRecord,
    MCPOAuthPendingAuthorization,
    MCPOAuthRepository,
    MCPToolListing,
    MCPUserIntegrationRecord,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


@dataclass(frozen=True)
class MCPOAuthConnectionState:
    server_name: str
    tokens: Optional[OAuthToken] = None
    pending: Optional[MCPOAuthPendingAuthorization] = None
    tool_listing: Optional[MCPToolListing] = None
    error: Optional[str] = None
    failed_at: Optional[datetime] = None
    disabled: bool = False
    tool_permissions: Dict[str, str] = field(default_factory=dict)


def _pending_expired(pending: Optional[MCPOAuthPendingAuthorization]) -> bool:
    return bool(pending and pending.expires_at < datetime.now(timezone.utc))


def _is_empty(record: MCPUserIntegrationRecord) -> bool:
    return (
        record.tokens is None
        and record.pending is None
        and record.tool_listing is None
        and record.error is None
        and not record.disabled
        and not record.tool_permissions
    )


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
            if not _is_empty(record):
                states[record.server_name] = MCPOAuthConnectionState(
                    server_name=record.server_name,
                    tokens=record.tokens,
                    pending=record.pending,
                    tool_listing=record.tool_listing,
                    error=record.error,
                    failed_at=record.failed_at,
                    disabled=record.disabled,
                    tool_permissions=dict(record.tool_permissions),
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
        # A new authorization is a new grant: the old one was refused (or is
        # being replaced), and what the new one lists may differ.
        record.tokens = None
        record.tool_listing = None
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
        record.tool_listing = None
        if _is_empty(record):
            await self.__repository.delete_user_integration(
                tenant, user_uuid, server_name
            )
            return
        await self.__repository.save_user_integration(record)

    async def __record(
        self, tenant: str, user_uuid: str, server_name: str
    ) -> MCPUserIntegrationRecord:
        return await self.__repository.get_user_integration(
            tenant, user_uuid, server_name
        ) or MCPUserIntegrationRecord(
            tenant=tenant, user_uuid=user_uuid, server_name=server_name
        )

    async def save_tool_listing(
        self,
        tenant: str,
        user_uuid: str,
        server_name: str,
        tool_listing: MCPToolListing,
    ) -> None:
        record = await self.__record(tenant, user_uuid, server_name)
        record.tool_listing = tool_listing
        record.error = None
        record.failed_at = None
        await self.__repository.save_user_integration(record)

    async def record_failure(
        self, tenant: str, user_uuid: str, server_name: str, error: str
    ) -> None:
        record = await self.__record(tenant, user_uuid, server_name)
        record.error = error
        record.failed_at = datetime.now(timezone.utc)
        await self.__repository.save_user_integration(record)

    async def set_disabled(
        self, tenant: str, user_uuid: str, server_name: str, disabled: bool
    ) -> None:
        record = await self.__record(tenant, user_uuid, server_name)
        record.disabled = disabled
        if _is_empty(record):
            await self.__repository.delete_user_integration(
                tenant, user_uuid, server_name
            )
            return
        await self.__repository.save_user_integration(record)

    async def set_tool_permissions(
        self,
        tenant: str,
        user_uuid: str,
        server_name: str,
        tool_permissions: Dict[str, str],
    ) -> None:
        record = await self.__record(tenant, user_uuid, server_name)
        record.tool_permissions = dict(tool_permissions)
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
