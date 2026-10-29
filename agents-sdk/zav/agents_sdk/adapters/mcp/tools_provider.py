import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol, Set

from mcp.types import Tool as MCPTool
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field, ValidationError

from zav.agents_sdk.adapters._filtering import is_source_active, passes_name_filter
from zav.agents_sdk.adapters.mcp.connection_service import (
    MCPConnectionService,
    MCPServerConnection,
    MCPServerUnavailableError,
)
from zav.agents_sdk.adapters.mcp.oauth_token_storage import MCPOAuthIntegrationStore
from zav.agents_sdk.adapters.mcp.server_config import (
    MCPServerConfig,
    server_config_from,
)
from zav.agents_sdk.adapters.mcp.strict_schema import ensure_strict_json_schema
from zav.agents_sdk.adapters.tool_policy import (
    ask_first,
    load_on_demand,
    waive_asking,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.mcp_configuration_store import (
    MCPConfigurationStore,
)
from zav.agents_sdk.domain.mcp_oauth import MCP_OAUTH_CONNECT_TOOL_NAME_PREFIX
from zav.agents_sdk.domain.request_headers import RequestHeaders
from zav.agents_sdk.domain.tools import (
    Tool,
    ToolDisplay,
    ToolNamespace,
    ToolOptions,
    ToolOrigin,
    ToolStreamingConfig,
)

# MCP tool names are only unique within a single server, but the agent flattens
# every provider's tools into one name-keyed registry. Qualifying each tool with
# its server keeps names globally unique so two servers can expose the same tool
# name without one silently overwriting the other.
MCP_TOOL_NAME_SEPARATOR = "__"
# Upper bound for cross-provider compatibility: the Anthropic API allows 128 but
# OpenAI function names must be <= 64 chars matching [A-Za-z0-9_-]. The separator
# stays within this charset so the qualified name is always a valid tool name.
MCP_TOOL_NAME_MAX_LENGTH = 64


class MCPMount(BaseModel):
    """A named MCP configuration the agent mounts, narrowed for this agent."""

    name: str = Field(
        ...,
        description=(
            "A named MCP configuration (tenant settings, or mcp_configurations.json "
            "in a local project)."
        ),
    )
    include_tools: Optional[List[str]] = Field(
        None, description="The server's tools this agent may call (its own names)."
    )
    exclude_tools: Optional[List[str]] = Field(
        None, description="The server's tools this agent may not call."
    )
    defer_tools: Optional[bool] = Field(
        None,
        description=(
            "Load this server's tools on demand through tool search. Unset "
            "follows the MCP tools provider's defer_tools."
        ),
    )


class MCPToolsProviderConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description="Enable MCP tools provider. If false, no tools will be provided.",
    )
    client_session_timeout_seconds: float = Field(
        2,
        description=(
            "Read timeout (seconds) for the MCP ClientSession, applied to the "
            "setup RPCs (initialize + list_tools). Kept short so a hanging or slow "
            "server fails fast during connect instead of blocking "
            "time-to-first-token. Tool calls use tool_call_timeout_seconds instead."
        ),
    )
    tool_call_timeout_seconds: float = Field(
        30,
        description=(
            "Read timeout (seconds) for individual MCP tool calls, passed per-call "
            "so real tool operations are not bounded by the short setup timeout."
        ),
    )
    convert_schemas_to_strict: bool = Field(
        False, description="Convert input schemas to strict JSON Schema."
    )
    servers: List[MCPServerConfig] = Field(
        default_factory=list,
        description="Inline server definitions. Prefer mounts.",
    )
    mounts: List[MCPMount] = Field(
        default_factory=list,
        description="The named MCP configurations this agent uses.",
    )
    defer_tools: bool = Field(
        False,
        description=(
            "Load these servers' tools on demand through tool search: the model "
            "sees each server by name and description. A mount's own "
            "defer_tools overrides this; the agent's tool search methods say how."
        ),
    )
    tool_streaming: Optional[Dict[str, ToolStreamingConfig]] = Field(
        default=None,
        description="Optional mapping of tool names to streaming configurations. "
        "If not provided, tools not in this mapping get auto-generated streaming text. "
        "If provided, only tools in this mapping get streamed.",
    )
    include_servers: Optional[List[str]] = Field(
        None, description="Allowlist of MCP server names to include."
    )
    exclude_servers: Optional[List[str]] = Field(
        None, description="Denylist of MCP server names to exclude."
    )
    selected_servers: Optional[List[str]] = Field(
        None,
        description=(
            "The servers a user picked for their agent (a request's or a user "
            "agent's `bot_params`). It only narrows the agent's servers; a name "
            "the agent does not have is ignored."
        ),
    )
    include_tools: Optional[List[str]] = Field(
        None, description="Allowlist of (namespaced) tool names to include."
    )
    exclude_tools: Optional[List[str]] = Field(
        None, description="Denylist of (namespaced) tool names to exclude."
    )
    confirmation_default: Literal["non_read_only", "all", "none"] = Field(
        "none",
        description=(
            "Which tools ask the user before each call, as a default a setting "
            "rule or the user may waive. `none` (default) asks for none; "
            "`non_read_only` asks for every tool the server does not annotate as "
            "read-only; `all` asks for every tool. A setting rule makes a "
            "specific tool always or never ask."
        ),
    )


@dataclass
class MCPToolEntry:
    """A tool of a connected server that the agent may call."""

    name: str
    # The name the agent calls it by (server prefix, shortened when long).
    qualified_name: str
    title: Optional[str]
    description: str
    read_only: bool


class MCPArgumentResolver(Protocol):
    """Fills what a model cannot write into an MCP tool call's arguments, such
    as a file's bytes, right before the call."""

    async def resolve(
        self, server_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]: ...


def qualified_tool_name(server_name: str, tool_name: str) -> str:
    name = f"{server_name}{MCP_TOOL_NAME_SEPARATOR}{tool_name}"
    if len(name) <= MCP_TOOL_NAME_MAX_LENGTH:
        return name
    # Over the limit: keep the tool name intact (the model reasons about it)
    # and shrink the server prefix so the qualified name fits.
    server_budget = (
        MCP_TOOL_NAME_MAX_LENGTH - len(MCP_TOOL_NAME_SEPARATOR) - len(tool_name)
    )
    if server_budget >= 1:
        return f"{server_name[:server_budget]}{MCP_TOOL_NAME_SEPARATOR}{tool_name}"
    # The tool name alone fills the budget; drop the server prefix.
    return tool_name[:MCP_TOOL_NAME_MAX_LENGTH]


# Names this provider's configuration as the reason a tool asks first.
_SETTINGS_SOURCE = "agent_mcp"


def mcp_tool_options(
    configuration: MCPToolsProviderConfiguration,
    read_only: Optional[bool],
    server_name: Optional[str] = None,
) -> ToolOptions:
    """Asking first and loading for an MCP tool, as the agent's MCP
    configuration and the server's mount say: ``confirmation_default`` asks
    only where no other rule decides."""
    options = ToolOptions()
    if configuration.confirmation_default == "all" or (
        # Annotations are server-supplied hints: a missing hint asks.
        configuration.confirmation_default == "non_read_only"
        and read_only is not True
    ):
        options = ask_first(options, _SETTINGS_SOURCE, required=False)
    elif configuration.confirmation_default == "non_read_only":
        options = waive_asking(options, _SETTINGS_SOURCE, default=True)
    mount = next(
        (mount for mount in configuration.mounts if mount.name == server_name), None
    )
    if (
        configuration.defer_tools
        if mount is None or mount.defer_tools is None
        else mount.defer_tools
    ):
        options = load_on_demand(options, True)
    return options


async def resolve_mcp_servers(
    configuration: MCPToolsProviderConfiguration,
    tenant: str,
    mcp_configuration_store: Optional[MCPConfigurationStore],
) -> List[MCPServerConfig]:
    """The agent's active servers: inline definitions and named
    configurations, narrowed by its server filters and the user's selection."""
    servers: List[MCPServerConfig] = list(configuration.servers)
    for name in [mount.name for mount in configuration.mounts]:
        named = (
            await mcp_configuration_store.get(tenant, name)
            if mcp_configuration_store
            else None
        )
        if named is None:
            logger.warning(f"MCP configuration '{name}' is not configured.")
            continue
        try:
            servers.append(server_config_from(name, named))
        except ValidationError as error:
            logger.warning(f"MCP configuration '{name}' is invalid: {error}")
    include = set(configuration.include_servers or []) or None
    exclude = set(configuration.exclude_servers or []) or None
    selected = configuration.selected_servers
    return [
        server_cfg
        for server_cfg in servers
        if is_source_active(
            server_cfg.get_transport_config().name, True, include, exclude
        )
        and (selected is None or server_cfg.get_transport_config().name in selected)
    ]


def granted_mcp_tools(
    configuration: MCPToolsProviderConfiguration,
    server_cfg: MCPServerConfig,
    mcp_tools: List[MCPTool],
) -> Set[str]:
    """The server's own tool names the agent may call, under the agent's, its
    mount's and the server's tool filters."""
    server_name = server_cfg.get_transport_config().name
    mount = next(
        (mount for mount in configuration.mounts if mount.name == server_name), None
    )
    # The agent's and the server's lists name namespaced tools; a mount's
    # names the server's own.
    filters = [
        (configuration.include_tools, configuration.exclude_tools, True),
        (server_cfg.include_tools, server_cfg.exclude_tools, True),
    ]
    if mount is not None:
        filters.append((mount.include_tools, mount.exclude_tools, False))
    return {
        mcp_tool.name
        for mcp_tool in mcp_tools
        if all(
            passes_name_filter(
                (
                    qualified_tool_name(server_name, mcp_tool.name)
                    if qualified
                    else mcp_tool.name
                ),
                set(include) if include else None,
                set(exclude) if exclude else None,
            )
            for include, exclude, qualified in filters
        )
    }


def mcp_tool_entries(
    configuration: MCPToolsProviderConfiguration,
    server_cfg: MCPServerConfig,
    connection: MCPServerConnection,
) -> List[MCPToolEntry]:
    """The listed tools of a server that the agent may call."""
    server_name = server_cfg.get_transport_config().name
    granted = granted_mcp_tools(configuration, server_cfg, connection.tools)
    entries: List[MCPToolEntry] = []
    for mcp_tool in connection.tools:
        if mcp_tool.name not in granted:
            continue
        tool_name = qualified_tool_name(server_name, mcp_tool.name)
        entries.append(
            MCPToolEntry(
                name=mcp_tool.name,
                qualified_name=tool_name,
                title=mcp_tool.title
                or (mcp_tool.annotations.title if mcp_tool.annotations else None),
                description=mcp_tool.description or "",
                read_only=bool(
                    mcp_tool.annotations and mcp_tool.annotations.read_only_hint
                ),
            )
        )
    return entries


class MCPToolsProvider:
    """The agent's MCP servers as tools: a connected server's listing becomes
    its tools, any other server a tool that connects it."""

    def __init__(
        self,
        mcp_tools_provider_configuration: MCPToolsProviderConfiguration,
        connection_service: MCPConnectionService,
        tenant: str,
        mcp_configuration_store: Optional[MCPConfigurationStore] = None,
        argument_resolver: Optional[MCPArgumentResolver] = None,
    ):
        self.__config = mcp_tools_provider_configuration
        self.__connection_service = connection_service
        self.__tenant = tenant
        self.__mcp_configuration_store = mcp_configuration_store
        self.__argument_resolver = argument_resolver
        self.__resolved: Optional[Dict[str, List[Tool]]] = None
        connection_service.on_listing_changed(self.__on_listing_changed)

    async def describe_loaded(self) -> Dict[str, Any]:
        resolved = await self.__resolve_server_tools()
        return {
            "enabled": self.__config.enabled,
            "sources": list(resolved.keys()),
            "tools_by_source": {
                name: [t.name for t in tools] for name, tools in resolved.items()
            },
        }

    async def get_tools(self) -> List[Tool]:
        """The servers' tools as last listed: read from the stored listings on
        the first call, then from memory. A connect or a server's tools/list
        changed notification updates them; no call here contacts a server."""
        if not self.__config.enabled:
            return []
        resolved = await self.__resolve_server_tools()
        return [t for tools in resolved.values() for t in tools]

    async def __on_listing_changed(
        self, server_cfg: MCPServerConfig, mcp_tools: List[MCPTool]
    ) -> None:
        server_name = server_cfg.get_transport_config().name
        if self.__resolved is not None and server_name in self.__resolved:
            self.__resolved[server_name] = await self.__build_server_tools(
                server_cfg, mcp_tools
            )

    async def cleanup(self):
        await self.__connection_service.cleanup()

    async def __resolve_server_tools(self) -> Dict[str, List[Tool]]:
        if self.__resolved is not None:
            return self.__resolved
        servers = await resolve_mcp_servers(
            self.__config,
            self.__tenant,
            self.__mcp_configuration_store,
        )
        connections = await self.__connection_service.connections(servers)
        resolved: Dict[str, List[Tool]] = {}
        for server_cfg in servers:
            server_name = server_cfg.get_transport_config().name
            connection = connections[server_name]
            if connection.status == "disabled":
                continue
            resolved[server_name] = (
                await self.__build_server_tools(server_cfg, connection.tools)
                if connection.status == "connected"
                else [self.__connect_tool(server_cfg)]
            )
        if connections:
            logger.info(
                "MCP servers: "
                + ", ".join(
                    f"{name}={connection.status}"
                    for name, connection in sorted(connections.items())
                )
            )
        self.__resolved = resolved
        return resolved

    def __authorization_required_response(
        self, server_name: str, failure_reason: str
    ) -> Dict[str, Any]:
        if self.__connection_service.authorization_url(server_name):
            return {
                "status": "authorization_required",
                "server_name": server_name,
                "display_text": f"{server_name} needs authorization.",
                "message": (
                    f"The {server_name} connection requires user authorization. "
                    "Ask the user to authorize the connection using the action "
                    "shown in the interface, then retry after they confirm "
                    "authorization is complete."
                ),
            }
        logger.warning(
            f"MCP server '{server_name}' connection unavailable: {failure_reason}"
        )
        return {
            "status": "connection_unavailable",
            "server_name": server_name,
            "display_text": f"{server_name} connection unavailable.",
            "message": (
                f"The {server_name} connection flow could not be started. "
                "Tell the user the connection is unavailable right now."
            ),
            "reason": failure_reason,
        }

    def __make_auth_aware_transform(self, server_name: str):
        def transform(response: Optional[Dict]) -> Optional[Dict]:
            if (
                isinstance(response, dict)
                and response.get("status") == "authorization_required"
            ):
                visible: Dict[str, Any] = {
                    "status": "authorization_required",
                    "server_name": response.get("server_name", server_name),
                    "display_text": response.get("display_text"),
                }
                auth_url = self.__connection_service.authorization_url(server_name)
                if auth_url:
                    visible["authorization_url"] = auth_url
                    visible["action_label"] = f"Authorize {server_name}"
                return visible
            return response

        return transform

    def __connect_tool(self, server_cfg: MCPServerConfig) -> Tool:
        """The tool through which the model connects a server the user has not
        connected yet; its tools join the turn once connected."""
        server_name = server_cfg.get_transport_config().name

        async def connect(**kwargs: Any) -> Dict[str, Any]:
            result = await self.__connection_service.connect(server_cfg)
            if result.status == "connected":
                tools = await self.__build_server_tools(server_cfg, result.tools)
                if self.__resolved is not None:
                    self.__resolved[server_name] = tools
                return {
                    "status": "connected",
                    "server_name": server_name,
                    "display_text": f"{server_name} connected: {len(tools)} tools",
                    "tools": [tool.name for tool in tools],
                    "message": (
                        f"{server_name} is connected. Its tools are now available; "
                        "use them to continue."
                    ),
                }
            return self.__authorization_required_response(
                server_name, result.error or "connection error"
            )

        def transform(response: Optional[Dict]) -> Optional[Dict]:
            if response is None:
                return None
            visible_response: Dict[str, Any] = {
                "server_name": server_name,
                "status": response.get("status"),
                "display_text": response.get("display_text"),
            }
            auth_url = self.__connection_service.authorization_url(server_name)
            if response.get("status") == "authorization_required" and auth_url:
                visible_response["authorization_url"] = auth_url
                visible_response["action_label"] = f"Authorize {server_name}"
            if response.get("reason"):
                visible_response["reason"] = response["reason"]
            if response.get("tools"):
                visible_response["tools"] = response["tools"]
            return visible_response

        return Tool(
            name=f"{MCP_OAUTH_CONNECT_TOOL_NAME_PREFIX}{server_name}",
            description=(
                f"Call this when the user asks to connect, use, or check the "
                f"status of {server_name}. Returns the connection status and "
                "whether user authorization is required."
            ),
            executable=connect,
            parameters_spec={"type": "object", "properties": {}, "required": []},
            streaming_config=ToolStreamingConfig(
                running_text=f"Checking {server_name} connection...",
                completed_text="{{ display_text }}",
                response_transform=transform,
            ),
            display=ToolDisplay(
                app=(server_cfg.display.app if server_cfg.display else None)
                or server_name,
                icon_url=server_cfg.display.icon_url if server_cfg.display else None,
            ),
        )

    async def __build_server_tools(
        self, server_cfg: MCPServerConfig, mcp_tools: List[MCPTool]
    ) -> List[Tool]:
        server_name = server_cfg.get_transport_config().name
        namespace = ToolNamespace(
            name=server_name,
            description=server_cfg.description
            or (server_cfg.display.app if server_cfg.display else None)
            or f"Tools of the {server_name} MCP server.",
        )
        granted = granted_mcp_tools(self.__config, server_cfg, mcp_tools)
        server_tools: List[Tool] = []
        for mcp_tool in mcp_tools:
            if mcp_tool.name not in granted:
                continue
            tool_name = qualified_tool_name(server_name, mcp_tool.name)
            read_only = bool(
                mcp_tool.annotations and mcp_tool.annotations.read_only_hint
            )

            schema = dict(mcp_tool.input_schema or {})
            if "properties" not in schema:
                schema["properties"] = {}
            if self.__config.convert_schemas_to_strict:
                try:
                    schema = ensure_strict_json_schema(schema)
                except Exception as e:
                    logger.info(f"Error converting MCP schema to strict mode: {e}")

            streaming_config: ToolStreamingConfig | None = None
            if self.__config.tool_streaming is None:
                fallback = f"Completed {tool_name}"
                streaming_config = ToolStreamingConfig(
                    running_text=f"Running {tool_name}...",
                    completed_text="{{ display_text or '" + fallback + "' }}",
                    response_transform=self.__make_auth_aware_transform(server_name),
                )
            else:
                streaming_config = self.__config.tool_streaming.get(tool_name)

            server_tools.append(
                Tool(
                    name=tool_name,
                    description=mcp_tool.description or "",
                    executable=self.__make_invoke(server_cfg, mcp_tool.name),
                    parameters_spec=schema,
                    streaming_config=streaming_config,
                    origin=ToolOrigin(
                        provider="mcp", source=server_name, name=mcp_tool.name
                    ),
                    display=self.__tool_display(server_cfg, mcp_tool),
                    namespace=namespace,
                    read_only=read_only,
                    options=mcp_tool_options(self.__config, read_only, server_name),
                )
            )
        return server_tools

    def __tool_display(
        self, server_cfg: MCPServerConfig, mcp_tool: MCPTool
    ) -> ToolDisplay:
        annotations = mcp_tool.annotations
        configured = server_cfg.display or ToolDisplay()
        return ToolDisplay(
            app=configured.app or server_cfg.get_transport_config().name,
            icon_url=configured.icon_url,
            title=mcp_tool.title or (annotations.title if annotations else None),
        )

    def __make_invoke(self, server_cfg: MCPServerConfig, tool_name: str):
        server_name = server_cfg.get_transport_config().name

        async def invoke(**kwargs: Any) -> Any:
            arguments = (
                await self.__argument_resolver.resolve(server_name, kwargs)
                if self.__argument_resolver is not None
                else kwargs
            )
            try:
                result = await self.__connection_service.call(
                    server_cfg, tool_name, arguments
                )
            except MCPServerUnavailableError as e:
                return self.__authorization_required_response(server_name, str(e))
            items = result.content or []
            if len(items) == 1:
                return items[0].model_dump_json()
            return json.dumps(
                {"parts": [item.model_dump(mode="json") for item in items]}
            )

        return invoke


class MCPConnectionServiceFactory(AgentDependencyFactory):
    # One service per agent: the provider and the connect tool share its sessions.
    __singleton__ = True

    @classmethod
    def create(
        cls,
        request_headers: RequestHeaders,
        mcp_oauth_store: MCPOAuthIntegrationStore,
        tenant: str = "zetaalpha",
        mcp_tools_provider_configuration: MCPToolsProviderConfiguration = (
            MCPToolsProviderConfiguration()
        ),
    ) -> MCPConnectionService:
        return MCPConnectionService(
            tenant=tenant,
            user_uuid=request_headers.requester_uuid,
            connection_store=mcp_oauth_store,
            client_session_timeout_seconds=(
                mcp_tools_provider_configuration.client_session_timeout_seconds
            ),
            tool_call_timeout_seconds=(
                mcp_tools_provider_configuration.tool_call_timeout_seconds
            ),
        )


class MCPToolsProviderFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(
        cls,
        mcp_connection_service: MCPConnectionService,
        mcp_argument_resolver: MCPArgumentResolver,
        tenant: str = "zetaalpha",
        mcp_tools_provider_configuration: MCPToolsProviderConfiguration = (
            MCPToolsProviderConfiguration()
        ),
        mcp_configuration_store: Optional[MCPConfigurationStore] = None,
    ) -> MCPToolsProvider:
        return MCPToolsProvider(
            mcp_tools_provider_configuration,
            connection_service=mcp_connection_service,
            tenant=tenant,
            mcp_configuration_store=mcp_configuration_store,
            argument_resolver=mcp_argument_resolver,
        )
