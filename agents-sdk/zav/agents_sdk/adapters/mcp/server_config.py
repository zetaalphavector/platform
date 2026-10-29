import hashlib
import json
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional, Union

from mcp.shared.auth import OAuthClientMetadata
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.tools import ToolDisplay


class OauthClientTransportConfig(BaseModel):
    server_url: str
    client_metadata: OAuthClientMetadata
    client_id: Optional[str] = Field(
        None,
        description="Pre-registered OAuth client ID. When provided together with "
        "client_secret, the SDK skips dynamic client registration (RFC 7591) "
        "and uses these credentials directly.",
    )
    client_secret: Optional[str] = Field(
        None,
        description="Pre-registered OAuth client secret.",
    )
    authorization_server_url: Optional[str] = Field(
        None,
        description="Authorization server to sign users in with, overriding the "
        "one the MCP server advertises (e.g. Entra for a cluster-internal server "
        "that names itself). When set, no `resource` parameter is sent.",
    )


class BaseTransportConfig(BaseModel):
    name: str = Field(..., description="Logical name for this MCP server.")
    oauth_client: Optional[OauthClientTransportConfig] = Field(
        None,
        description="Optional OAuth client configuration for this server.",
    )


class StdIoTransportConfig(BaseTransportConfig):

    command: str = Field(..., description="Command to launch the server.")
    args: List[str] = Field(
        default_factory=list, description="Arguments to pass to the command."
    )
    env: Optional[Dict[str, str]] = Field(
        None, description="Optional environment variables for the server process."
    )
    cwd: Optional[str] = Field(
        None, description="Optional working directory for the server process."
    )
    encoding: str = Field(
        "utf-8",
        description="Optional encoding for the server process to send and receive"
        " messages. Defaults to UTF-8.",
    )
    encoding_error_handler: Literal["strict", "replace", "ignore"] = Field(
        "strict",
        description="Optional error handler for encoding errors. Defaults to 'strict'.",
    )


class SseTransportConfig(BaseTransportConfig):

    url: str = Field(..., description="URL of the MCP server")
    headers: Optional[Dict[str, str]] = Field(
        None, description="Optional HTTP headers sent to the server."
    )
    timeout: float = Field(
        5,
        description="Optional timeout for HTTP requests (in seconds)."
        " Defaults to 5 seconds.",
    )
    sse_read_timeout: float = Field(
        300,
        description="Optional timeout for SSE read operations (in seconds)."
        " Defaults to 5 minutes.",
    )


class StreamableHttpTransportConfig(BaseTransportConfig):

    url: str = Field(..., description="URL of the MCP server")
    headers: Optional[Dict[str, str]] = Field(
        None, description="Optional HTTP headers sent to the server."
    )
    timeout: timedelta = Field(
        timedelta(seconds=5),
        description="Optional timeout for HTTP requests (in seconds)."
        " Defaults to 5 seconds.",
    )
    sse_read_timeout: timedelta = Field(
        timedelta(seconds=300),
        description="Optional timeout for SSE read operations (in seconds)."
        " Defaults to 5 minutes.",
    )
    terminate_on_close: bool = Field(
        True, description="Whether to terminate the connection when the client closes."
    )


class MCPServerTransportConfig(BaseModel):
    stdio: Optional[StdIoTransportConfig] = None
    sse: Optional[SseTransportConfig] = None
    streamable_http: Optional[StreamableHttpTransportConfig] = None


class MCPServerConfig(BaseModel):
    transport: Literal["stdio", "http-sse", "streamable-http"]
    transport_config: MCPServerTransportConfig
    display: Optional[ToolDisplay] = Field(
        None,
        description=(
            "App name and icon shown on the confirmation card of this server's "
            "tools. The card title always comes from the server's tool metadata."
        ),
    )
    description: Optional[str] = Field(
        None,
        description=(
            "What the server is for. The agent reads it to decide whether to "
            "connect the server and, with tool search, whether to load its tools."
        ),
    )
    include_tools: Optional[List[str]] = Field(
        None, description="Allowlist of this server's (namespaced) tool names."
    )
    exclude_tools: Optional[List[str]] = Field(
        None, description="Denylist of this server's (namespaced) tool names."
    )

    def get_transport_config(
        self,
    ) -> Union[
        StdIoTransportConfig,
        SseTransportConfig,
        StreamableHttpTransportConfig,
    ]:
        if self.transport == "stdio" and self.transport_config.stdio:
            return self.transport_config.stdio
        elif self.transport == "http-sse" and self.transport_config.sse:
            return self.transport_config.sse
        elif (
            self.transport == "streamable-http"
            and self.transport_config.streamable_http
        ):
            return self.transport_config.streamable_http
        else:
            raise ValueError(
                f"Unsupported transport type: {self.transport}. "
                "Expected 'stdio' or 'http-sse'."
            )


def server_fingerprint(server_cfg: MCPServerConfig) -> str:
    # What a saved tool list and the user's tokens were obtained from, sign-in
    # settings included: tokens from another authorization server, client or
    # scope do not fit the server. No secrets: they are not part of the
    # server's identity, and a configuration loads them freshly encrypted.
    transport_cfg = server_cfg.get_transport_config()
    oauth_cfg = transport_cfg.oauth_client
    if isinstance(transport_cfg, StdIoTransportConfig):
        endpoint: Any = [transport_cfg.command, *transport_cfg.args]
    else:
        endpoint = transport_cfg.url
    identity = {
        "transport": server_cfg.transport,
        "name": transport_cfg.name,
        "endpoint": endpoint,
        "oauth_server_url": oauth_cfg.server_url if oauth_cfg else None,
        "oauth_sign_in": (
            [
                oauth_cfg.authorization_server_url,
                oauth_cfg.client_id,
                oauth_cfg.client_metadata.scope,
            ]
            if oauth_cfg
            else None
        ),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _oauth_client_with_defaults(oauth_client: Dict[str, Any]) -> Dict[str, Any]:
    """An OAuth client block with the client metadata a named configuration may
    leave out: a pre-registered client sends its secret in the token request
    body, or no secret when it has none. Explicit client metadata wins."""
    client_metadata = dict(oauth_client.get("client_metadata") or {})
    if oauth_client.get("client_id"):
        client_metadata.setdefault(
            "token_endpoint_auth_method",
            "client_secret_post" if oauth_client.get("client_secret") else "none",
        )
    return {**oauth_client, "client_metadata": client_metadata}


def server_config_from(name: str, configuration: Dict[str, Any]) -> MCPServerConfig:
    """A named configuration as a server: its name is the server's name."""
    transport_config = {
        transport: {
            **settings,
            "name": name,
            **(
                {"oauth_client": _oauth_client_with_defaults(settings["oauth_client"])}
                if settings.get("oauth_client")
                else {}
            ),
        }
        for transport, settings in (configuration.get("transport_config") or {}).items()
        if settings
    }
    return MCPServerConfig.parse_obj(
        {
            key: value
            for key, value in configuration.items()
            if key
            in ("transport", "display", "description", "include_tools", "exclude_tools")
        }
        | {"transport_config": transport_config}
    )
