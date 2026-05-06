import asyncio
import json
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from mcp.types import Tool as MCPTool
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.mcp.strict_schema import ensure_strict_json_schema
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, ToolStreamingConfig


async def _async_redirect_handler(url: str) -> None:
    logger.info(f"OAuth: Visit this URL to authorize: {url}")


async def _async_callback_handler() -> tuple:
    return ("auth_code", None)


class InMemoryTokenStorage(TokenStorage):
    # Adapted from https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/clients/simple-auth-client/mcp_simple_auth_client/main.py # noqa: E501

    def __init__(self):
        self.__tokens: OAuthToken | None = None
        self.__client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self.__tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.__tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self.__client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.__client_info = client_info


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


class MCPToolsProviderConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description="Enable MCP tools provider. If false, no tools will be provided.",
    )
    client_session_timeout_seconds: float = Field(
        5, description="Read timeout in seconds passed to the MCP ClientSession."
    )
    convert_schemas_to_strict: bool = Field(
        False, description="Convert input schemas to strict JSON Schema."
    )
    servers: List[MCPServerConfig] = Field(default_factory=list)
    tool_streaming: Optional[Dict[str, ToolStreamingConfig]] = Field(
        default=None,
        description="Optional mapping of tool names to streaming configurations. "
        "If not provided, tools not in this mapping get auto-generated streaming text. "
        "If provided, only tools in this mapping get streamed.",
    )


class MCPToolsProvider:

    def __init__(self, mcp_tools_provider_configuration: MCPToolsProviderConfiguration):
        self.__config = mcp_tools_provider_configuration
        self.__stack = AsyncExitStack()
        self.__lock = asyncio.Lock()
        self.__session_map: Dict[str, ClientSession] = {}
        self.__resolved: Optional[Dict[str, List[Tool]]] = None

    async def describe_loaded(self) -> Dict[str, Any]:
        resolved = await self.__resolve_server_tools()
        return {
            "enabled": self.__config.enabled,
            "sources": list(resolved.keys()),
            "tools_by_source": {
                name: [t.name for t in tools] for name, tools in resolved.items()
            },
        }

    async def cleanup(self):
        async with self.__lock:
            try:
                await self.__stack.aclose()
            except Exception as e:
                logger.error(f"Error cleaning up server: {e}")
            finally:
                self.session = None
                self.__session_map = {}

    async def __connect(self):
        try:
            for server_cfg in self.__config.servers:
                transport_cfg = server_cfg.get_transport_config()
                oauth_auth = None
                if oauth_cfg := transport_cfg.oauth_client:
                    storage = InMemoryTokenStorage()
                    if oauth_cfg.client_id:
                        await storage.set_client_info(
                            OAuthClientInformationFull(
                                client_id=oauth_cfg.client_id,
                                client_secret=oauth_cfg.client_secret,
                                **oauth_cfg.client_metadata.model_dump(
                                    exclude_none=True
                                ),
                            )
                        )
                    oauth_auth = OAuthClientProvider(
                        server_url=oauth_cfg.server_url,
                        client_metadata=OAuthClientMetadata.model_validate(
                            oauth_cfg.client_metadata
                        ),
                        storage=storage,
                        redirect_handler=lambda url: _async_redirect_handler(url),
                        # TODO: Implement telling the user to visit the URL as an
                        # agent tool response
                        callback_handler=lambda: _async_callback_handler(),
                        # TODO: Implement getting auth code from callback endpoint
                    )
                if isinstance(transport_cfg, StdIoTransportConfig):
                    transport = await self.__stack.enter_async_context(
                        stdio_client(
                            StdioServerParameters(
                                command=transport_cfg.command,
                                args=transport_cfg.args,
                                env=transport_cfg.env,
                                cwd=transport_cfg.cwd,
                                encoding=transport_cfg.encoding,
                                encoding_error_handler=(
                                    transport_cfg.encoding_error_handler
                                ),
                            )
                        )
                    )
                elif isinstance(transport_cfg, SseTransportConfig):
                    transport = await self.__stack.enter_async_context(
                        sse_client(
                            url=transport_cfg.url,
                            headers=transport_cfg.headers,
                            timeout=transport_cfg.timeout,
                            sse_read_timeout=transport_cfg.sse_read_timeout,
                            auth=oauth_auth,
                        )
                    )
                elif isinstance(transport_cfg, StreamableHttpTransportConfig):
                    transport = await self.__stack.enter_async_context(
                        streamablehttp_client(
                            url=transport_cfg.url,
                            headers=transport_cfg.headers,
                            timeout=transport_cfg.timeout,
                            sse_read_timeout=transport_cfg.sse_read_timeout,
                            terminate_on_close=transport_cfg.terminate_on_close,
                            auth=oauth_auth,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported transport: {server_cfg.transport}")
                read, write, *_ = transport
                session = await self.__stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        read_timeout_seconds=(
                            timedelta(
                                seconds=self.__config.client_session_timeout_seconds
                            )
                            if self.__config.client_session_timeout_seconds
                            else None
                        ),
                    )
                )
                await session.initialize()
                self.__session_map[transport_cfg.name] = session
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            await self.__stack.aclose()
            raise

    async def get_tools(self) -> List[Tool]:
        """
        Discover MCP tools from all configured servers and
        return them as Tool instances.
        """
        if not self.__config.enabled:
            return []
        resolved = await self.__resolve_server_tools()
        return [t for tools in resolved.values() for t in tools]

    async def __resolve_server_tools(
        self,
    ) -> Dict[str, List[Tool]]:
        if self.__resolved is not None:
            return self.__resolved
        if not self.__session_map:
            await self.__connect()
        self.__resolved = {}
        for server_cfg in self.__config.servers:
            transport_cfg = server_cfg.get_transport_config()
            server_name = transport_cfg.name
            server_tools: List[Tool] = []
            mcp_tools = await self.__session_map[server_name].list_tools()
            for mcp_tool in mcp_tools.tools:
                schema = dict(mcp_tool.inputSchema or {})
                if "properties" not in schema:
                    schema["properties"] = {}
                # Convert schema to strict JSON Schema if requested
                if self.__config.convert_schemas_to_strict:
                    try:
                        schema = ensure_strict_json_schema(schema)
                    except Exception as e:
                        logger.info(f"Error converting MCP schema to strict mode: {e}")

                # Build an invoke function bound to this server and tool
                def make_invoke(s: ClientSession, t: MCPTool):
                    async def invoke(**kwargs: Any) -> str:
                        result = await s.call_tool(t.name, kwargs)
                        items = result.content or []
                        if len(items) == 1:
                            return items[0].model_dump_json()
                        return json.dumps(
                            {"parts": [item.model_dump(mode="json") for item in items]}
                        )

                    return invoke

                invoke_fn = make_invoke(self.__session_map[server_name], mcp_tool)

                streaming_config: ToolStreamingConfig | None = None
                if self.__config.tool_streaming is None:
                    streaming_config = ToolStreamingConfig(
                        running_text=f"Running {mcp_tool.name}...",
                        completed_text=f"Completed {mcp_tool.name}",
                    )
                else:
                    streaming_config = self.__config.tool_streaming.get(mcp_tool.name)

                server_tools.append(
                    Tool(
                        name=mcp_tool.name,
                        description=mcp_tool.description or "",
                        executable=invoke_fn,
                        parameters_spec=schema,
                        streaming_config=streaming_config,
                    )
                )
            self.__resolved[server_name] = server_tools
        return self.__resolved


class MCPToolsProviderFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        mcp_tools_provider_configuration: MCPToolsProviderConfiguration = (
            MCPToolsProviderConfiguration()
        ),
    ) -> MCPToolsProvider:
        return MCPToolsProvider(mcp_tools_provider_configuration)
