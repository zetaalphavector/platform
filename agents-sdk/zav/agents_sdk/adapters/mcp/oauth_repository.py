from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


@dataclass(frozen=True)
class MCPOAuthPendingAuthorization:
    state: str
    tenant: str
    server_name: str
    user_uuid: str
    auth_url: str
    code_verifier: str
    token_endpoint: str
    redirect_uri: str
    expires_at: datetime
    resource: Optional[str] = None


@dataclass(frozen=True)
class MCPToolListing:
    """A server's `tools/list`, as of `listed_at`, from the endpoint
    `fingerprint` identifies."""

    fingerprint: str
    listed_at: datetime
    tools: List[Dict[str, Any]]


@dataclass
class MCPUserIntegrationRecord:
    tenant: str
    user_uuid: str
    server_name: str
    pending: Optional[MCPOAuthPendingAuthorization] = None
    tokens: Optional[OAuthToken] = None
    tool_listing: Optional[MCPToolListing] = None
    # The last connect failure, cleared by the next success.
    error: Optional[str] = None
    failed_at: Optional[datetime] = None
    # The user turned the server off.
    disabled: bool = False
    # The user's choice per tool (the server's own tool name): "allow" runs it
    # without asking, "ask" asks every time, "off" hides it from their agents.
    tool_permissions: Dict[str, str] = field(default_factory=dict)


@dataclass
class MCPOAuthClientInfoRecord:
    tenant: str
    server_name: str
    client_info: OAuthClientInformationFull


class MCPOAuthRepository(ABC):
    @abstractmethod
    async def list_user_integrations(
        self, tenant: str, user_uuid: str, server_names: List[str]
    ) -> List[MCPUserIntegrationRecord]:
        raise NotImplementedError

    @abstractmethod
    async def get_user_integration(
        self, tenant: str, user_uuid: str, server_name: str
    ) -> Optional[MCPUserIntegrationRecord]:
        raise NotImplementedError

    @abstractmethod
    async def find_user_integration_by_pending_state(
        self, state: str
    ) -> Optional[MCPUserIntegrationRecord]:
        raise NotImplementedError

    @abstractmethod
    async def save_user_integration(self, record: MCPUserIntegrationRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete_user_integration(
        self, tenant: str, user_uuid: str, server_name: str
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def list_client_info(
        self, tenant: str, server_names: List[str]
    ) -> List[MCPOAuthClientInfoRecord]:
        raise NotImplementedError

    @abstractmethod
    async def get_client_info(
        self, tenant: str, server_name: str
    ) -> Optional[MCPOAuthClientInfoRecord]:
        raise NotImplementedError

    @abstractmethod
    async def save_client_info(self, record: MCPOAuthClientInfoRecord) -> None:
        raise NotImplementedError
