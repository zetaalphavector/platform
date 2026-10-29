import json
import os
from typing import Any, Dict, Optional, Protocol

from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.secret_store import SecretStore


class MCPConfigurationStore(Protocol):
    """Named MCP server configurations per tenant: `name`, `description`,
    `transport`, `transport_config`, `display` and the server's tool filters,
    credentials materialized. Project files for `za agents serve`, tenant
    settings for the hosted chat service."""

    async def get(self, tenant: str, name: str) -> Optional[Dict[str, Any]]: ...


class MCPConfigurationStoreFactory(AgentDependencyFactory):
    def __init__(self, mcp_configuration_store: MCPConfigurationStore):
        self.__mcp_configuration_store = mcp_configuration_store

    def create(self) -> MCPConfigurationStore:  # type: ignore[override]
        return self.__mcp_configuration_store


def materialize_mcp_configuration(
    configuration: Dict[str, Any], secret_store: Optional[SecretStore]
) -> Dict[str, Any]:
    """A configuration as a server definition: `headers_refs` become headers
    and `client_secret_ref` the OAuth client secret of its transport. A ref to
    a secret the store does not hold leaves that credential unset."""

    def secret(name: Any) -> Optional[str]:
        stored = (
            secret_store.get(name) if secret_store and isinstance(name, str) else None
        )
        return stored.get_unencrypted_secret() if stored is not None else None

    headers = {
        header: value
        for header, ref in (configuration.get("headers_refs") or {}).items()
        if (value := secret(ref)) is not None
    }
    client_secret = secret(configuration.get("client_secret_ref"))
    transport_config: Dict[str, Any] = {}
    for transport, settings in (configuration.get("transport_config") or {}).items():
        settings = dict(settings or {})
        if headers:
            settings["headers"] = {**(settings.get("headers") or {}), **headers}
        if client_secret is not None and settings.get("oauth_client") is not None:
            settings["oauth_client"] = {
                **settings["oauth_client"],
                "client_secret": client_secret,
            }
        transport_config[transport] = settings
    return {
        **{
            key: value
            for key, value in configuration.items()
            if key not in ("headers_refs", "client_secret_ref")
        },
        "transport_config": transport_config,
    }


class MCPConfigurationStoreFromFile:
    """mcp_configurations.json, a sibling of agent_setups.json, for every
    tenant; credentials reference env/secrets.json by name."""

    def __init__(
        self,
        agent_setups_path: Optional[str] = None,
        mcp_configurations_path: Optional[str] = None,
        secret_store: Optional[SecretStore] = None,
    ):
        if mcp_configurations_path is None and agent_setups_path:
            mcp_configurations_path = os.path.join(
                os.path.dirname(agent_setups_path), "mcp_configurations.json"
            )
        self.__configurations: Dict[str, Dict[str, Any]] = {}
        if mcp_configurations_path and os.path.isfile(mcp_configurations_path):
            with open(mcp_configurations_path, "r") as file:
                for configuration in json.load(file):
                    if configuration.get("name"):
                        self.__configurations[configuration["name"]] = configuration
        self.__secret_store = secret_store

    async def get(self, tenant: str, name: str) -> Optional[Dict[str, Any]]:
        configuration = self.__configurations.get(name)
        if configuration is None:
            return None
        return materialize_mcp_configuration(configuration, self.__secret_store)
