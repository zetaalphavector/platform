from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Type

from zav.agents_sdk.adapters.mcp import (
    MCPOAuthIntegrationStoreFactory,
    get_local_file_mcp_oauth_integration_store,
    get_mcp_oauth_token_client,
)

if TYPE_CHECKING:
    from zav.agents_sdk.adapters.mcp import (
        MCPOAuthIntegrationStore,
        MCPOAuthTokenClient,
    )
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry


def setup_local_mcp_oauth(
    agent_dependency_registry: Type[AgentDependencyRegistry],
) -> Tuple[MCPOAuthIntegrationStore, MCPOAuthTokenClient]:
    """Register the local-file MCPOAuthIntegrationStore and the token client in the
    dependency registry, and return them for use in the app setup."""
    mcp_oauth_store = get_local_file_mcp_oauth_integration_store()
    mcp_oauth_token_client = get_mcp_oauth_token_client()
    agent_dependency_registry.register(MCPOAuthIntegrationStoreFactory(mcp_oauth_store))
    return mcp_oauth_store, mcp_oauth_token_client
