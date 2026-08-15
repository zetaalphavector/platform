import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.mcp.oauth_repository import (
    MCPOAuthClientInfoRecord,
    MCPOAuthPendingAuthorization,
    MCPOAuthRepository,
    MCPUserIntegrationRecord,
)
from zav.agents_sdk.adapters.mcp.oauth_serialization import (
    oauth_token_expires_at,
    oauth_token_from_storage,
)
from zav.agents_sdk.adapters.mcp.oauth_token_storage import (
    MCPOAuthIntegrationStore,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class LocalFileMCPOAuthIntegrationStoreConfiguration(BaseModel):
    file_path: str = Field(
        ".mcp_oauth_integrations.json",
        description="Path to the local JSON file for MCP OAuth state.",
    )


def _integration_key(tenant: str, user_uuid: str, server_name: str) -> str:
    return f"{tenant}:{user_uuid}:{server_name}"


def _client_info_key(tenant: str, server_name: str) -> str:
    return f"{tenant}:{server_name}"


def _format_datetime(value: datetime) -> str:
    return value.isoformat()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _pending_to_data(pending: MCPOAuthPendingAuthorization) -> Dict[str, Any]:
    return {
        "state": pending.state,
        "tenant": pending.tenant,
        "server_name": pending.server_name,
        "user_uuid": pending.user_uuid,
        "auth_url": pending.auth_url,
        "code_verifier": pending.code_verifier,
        "token_endpoint": pending.token_endpoint,
        "redirect_uri": pending.redirect_uri,
        "resource": pending.resource,
        "expires_at": _format_datetime(pending.expires_at),
    }


def _pending_from_data(
    data: Optional[Dict[str, Any]],
) -> Optional[MCPOAuthPendingAuthorization]:
    if not data:
        return None
    return MCPOAuthPendingAuthorization(
        state=data["state"],
        tenant=data["tenant"],
        server_name=data["server_name"],
        user_uuid=data["user_uuid"],
        auth_url=data["auth_url"],
        code_verifier=data["code_verifier"],
        token_endpoint=data["token_endpoint"],
        redirect_uri=data["redirect_uri"],
        resource=data.get("resource"),
        expires_at=_parse_datetime(data["expires_at"]),
    )


def _tokens_to_data(tokens: OAuthToken) -> Dict[str, Any]:
    expires_at = oauth_token_expires_at(tokens)
    return {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "token_type": tokens.token_type,
        "scope": tokens.scope,
        "expires_at": _format_datetime(expires_at) if expires_at else None,
    }


def _tokens_from_data(data: Optional[Dict[str, Any]]) -> Optional[OAuthToken]:
    if not data:
        return None
    expires_at_str = data.get("expires_at")
    return oauth_token_from_storage(
        access_token=data.get("access_token"),
        refresh_token=data.get("refresh_token"),
        token_type=data.get("token_type"),
        scope=data.get("scope"),
        expires_at=_parse_datetime(expires_at_str) if expires_at_str else None,
    )


def _record_to_data(record: MCPUserIntegrationRecord) -> Dict[str, Any]:
    return {
        "tenant": record.tenant,
        "user_uuid": record.user_uuid,
        "server_name": record.server_name,
        "pending": _pending_to_data(record.pending) if record.pending else None,
        "tokens": _tokens_to_data(record.tokens) if record.tokens else None,
    }


def _record_from_data(
    data: Optional[Dict[str, Any]],
) -> Optional[MCPUserIntegrationRecord]:
    if data is None:
        return None
    return MCPUserIntegrationRecord(
        tenant=data["tenant"],
        user_uuid=data["user_uuid"],
        server_name=data["server_name"],
        pending=_pending_from_data(data.get("pending")),
        tokens=_tokens_from_data(data.get("tokens")),
    )


class LocalFileMCPOAuthRepository(MCPOAuthRepository):
    def __init__(self, file_path: str):
        self.__path = Path(file_path)
        self.__path.parent.mkdir(parents=True, exist_ok=True)

    async def list_user_integrations(
        self, tenant: str, user_uuid: str, server_names: List[str]
    ) -> List[MCPUserIntegrationRecord]:
        data = self.__read_data()
        records = []
        for server_name in server_names:
            record = _record_from_data(
                data["integrations"].get(
                    _integration_key(tenant, user_uuid, server_name)
                )
            )
            if record is not None:
                records.append(record)
        return records

    async def get_user_integration(
        self, tenant: str, user_uuid: str, server_name: str
    ) -> Optional[MCPUserIntegrationRecord]:
        data = self.__read_data()
        return _record_from_data(
            data["integrations"].get(_integration_key(tenant, user_uuid, server_name))
        )

    async def find_user_integration_by_pending_state(
        self, state: str
    ) -> Optional[MCPUserIntegrationRecord]:
        data = self.__read_data()
        for record_data in data["integrations"].values():
            pending = record_data.get("pending")
            if pending and pending.get("state") == state:
                return _record_from_data(record_data)
        return None

    async def save_user_integration(self, record: MCPUserIntegrationRecord) -> None:
        data = self.__read_data()
        key = _integration_key(record.tenant, record.user_uuid, record.server_name)
        data["integrations"][key] = _record_to_data(record)
        self.__write_data(data)

    async def delete_user_integration(
        self, tenant: str, user_uuid: str, server_name: str
    ) -> bool:
        data = self.__read_data()
        key = _integration_key(tenant, user_uuid, server_name)
        if key not in data["integrations"]:
            return False
        del data["integrations"][key]
        self.__write_data(data)
        return True

    async def list_client_info(
        self, tenant: str, server_names: List[str]
    ) -> List[MCPOAuthClientInfoRecord]:
        data = self.__read_data()
        records = []
        for server_name in server_names:
            key = _client_info_key(tenant, server_name)
            if key in data["client_info"]:
                records.append(
                    MCPOAuthClientInfoRecord(
                        tenant=tenant,
                        server_name=server_name,
                        client_info=OAuthClientInformationFull.model_validate(
                            data["client_info"][key]
                        ),
                    )
                )
        return records

    async def get_client_info(
        self, tenant: str, server_name: str
    ) -> Optional[MCPOAuthClientInfoRecord]:
        data = self.__read_data()
        key = _client_info_key(tenant, server_name)
        if key not in data["client_info"]:
            return None
        return MCPOAuthClientInfoRecord(
            tenant=tenant,
            server_name=server_name,
            client_info=OAuthClientInformationFull.model_validate(
                data["client_info"][key]
            ),
        )

    async def save_client_info(self, record: MCPOAuthClientInfoRecord) -> None:
        data = self.__read_data()
        key = _client_info_key(record.tenant, record.server_name)
        data["client_info"][key] = record.client_info.model_dump(mode="json")
        self.__write_data(data)

    def __read_data(self) -> Dict[str, Dict[str, Any]]:
        if not self.__path.exists():
            return {"client_info": {}, "integrations": {}}
        try:
            data = json.loads(self.__path.read_text())
        except json.JSONDecodeError:
            return {"client_info": {}, "integrations": {}}
        return {
            "client_info": data.get("client_info", {}),
            "integrations": data.get("integrations", {}),
        }

    def __write_data(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.__path.write_text(json.dumps(data, indent=2, sort_keys=True))


class LocalFileMCPOAuthIntegrationStoreFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        local_file_mcp_oauth_integration_store_configuration: (
            LocalFileMCPOAuthIntegrationStoreConfiguration
        ) = LocalFileMCPOAuthIntegrationStoreConfiguration(),
    ) -> MCPOAuthIntegrationStore:
        return MCPOAuthIntegrationStore(
            LocalFileMCPOAuthRepository(
                file_path=(
                    local_file_mcp_oauth_integration_store_configuration.file_path
                ),
            )
        )


def get_local_file_mcp_oauth_integration_store(
    configuration: Optional[LocalFileMCPOAuthIntegrationStoreConfiguration] = None,
) -> MCPOAuthIntegrationStore:
    if configuration is None:
        configuration = LocalFileMCPOAuthIntegrationStoreConfiguration()
    return MCPOAuthIntegrationStore(
        LocalFileMCPOAuthRepository(file_path=configuration.file_path)
    )
