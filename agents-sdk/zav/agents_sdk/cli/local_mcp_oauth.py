from __future__ import annotations

import os
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
from zav.agents_sdk.cli.project_config import MCP_CONFIGURATIONS_FILE
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry
from zav.agents_sdk.domain.mcp_configuration_store import (
    MCPConfigurationStoreFactory,
    MCPConfigurationStoreFromFile,
)
from zav.agents_sdk.domain.secret_store import SecretStore


def setup_local_mcp_oauth(
    agent_dependency_registry: Type[AgentDependencyRegistry],
) -> Tuple[MCPOAuthIntegrationStore, MCPOAuthTokenClient]:
    """Register the local-file MCPOAuthIntegrationStore and the token client in the
    dependency registry, and return them for use in the app setup."""
    mcp_oauth_store = get_local_file_mcp_oauth_integration_store()
    mcp_oauth_token_client = get_mcp_oauth_token_client()
    agent_dependency_registry.register(MCPOAuthIntegrationStoreFactory(mcp_oauth_store))
    return mcp_oauth_store, mcp_oauth_token_client


def setup_local_mcp_configurations(
    agent_dependency_registry: Type[AgentDependencyRegistry],
    project_dir: str,
    secret_store: SecretStore,
) -> None:
    """Register the project's mcp_configurations.json, so agents can use MCP
    servers by name; credentials reference env/secrets.json."""
    agent_dependency_registry.register(
        MCPConfigurationStoreFactory(
            MCPConfigurationStoreFromFile(
                mcp_configurations_path=os.path.join(
                    project_dir, MCP_CONFIGURATIONS_FILE
                ),
                secret_store=secret_store,
            )
        )
    )
