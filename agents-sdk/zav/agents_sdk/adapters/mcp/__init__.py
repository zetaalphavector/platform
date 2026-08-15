from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

__all__ = ["AgentDependencyRegistry"]

try:
    from zav.agents_sdk.adapters.mcp.local_oauth_repository import (
        LocalFileMCPOAuthIntegrationStoreConfiguration,
        LocalFileMCPOAuthIntegrationStoreFactory,
        LocalFileMCPOAuthRepository,
        get_local_file_mcp_oauth_integration_store,
    )
    from zav.agents_sdk.adapters.mcp.oauth_repository import (
        MCPOAuthClientInfoRecord,
        MCPOAuthPendingAuthorization,
        MCPOAuthRepository,
        MCPUserIntegrationRecord,
    )
    from zav.agents_sdk.adapters.mcp.oauth_token_client import (
        MCPOAuthTokenClient,
        MCPOAuthTokenExchangeError,
        get_mcp_oauth_token_client,
    )
    from zav.agents_sdk.adapters.mcp.oauth_token_storage import (
        MCPOAuthConnectionState,
        MCPOAuthIntegrationStore,
        MCPOAuthIntegrationStoreFactory,
        PersistentTokenStorage,
    )
    from zav.agents_sdk.adapters.mcp.tools_provider import (
        MCPServerConfig,
        MCPServerTransportConfig,
        MCPToolsProvider,
        MCPToolsProviderConfiguration,
        MCPToolsProviderFactory,
        SseTransportConfig,
        StdIoTransportConfig,
    )

    AgentDependencyRegistry.register(LocalFileMCPOAuthIntegrationStoreFactory)
    AgentDependencyRegistry.register(MCPToolsProviderFactory)
    __all__ += [
        "LocalFileMCPOAuthIntegrationStoreConfiguration",
        "LocalFileMCPOAuthIntegrationStoreFactory",
        "LocalFileMCPOAuthRepository",
        "get_local_file_mcp_oauth_integration_store",
        "MCPOAuthClientInfoRecord",
        "MCPOAuthRepository",
        "MCPOAuthTokenClient",
        "MCPOAuthTokenExchangeError",
        "MCPUserIntegrationRecord",
        "get_mcp_oauth_token_client",
        "MCPToolsProviderConfiguration",
        "MCPServerConfig",
        "MCPServerTransportConfig",
        "MCPOAuthConnectionState",
        "MCPOAuthIntegrationStore",
        "MCPOAuthPendingAuthorization",
        "MCPToolsProvider",
        "MCPToolsProviderFactory",
        "PersistentTokenStorage",
        "MCPOAuthIntegrationStoreFactory",
        "StdIoTransportConfig",
        "SseTransportConfig",
    ]
except ImportError:
    pass
