import asyncio
import json
import secrets
from contextlib import AsyncExitStack
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlencode, urlparse

import httpx2
from mcp import ClientSession, StdioServerParameters
from mcp.client.auth import (
    OAuthClientProvider,
    OAuthFlowError,
    PKCEParameters,
    TokenStorage,
)
from mcp.client.auth.oauth2 import OAuthContext
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_oauth_metadata_request,
    handle_auth_metadata_response,
    handle_protected_resource_response,
)
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from mcp.types import Tool as MCPTool
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.mcp.oauth_repository import MCPOAuthPendingAuthorization
from zav.agents_sdk.adapters.mcp.oauth_token_storage import (
    MCPOAuthConnectionState,
    MCPOAuthIntegrationStore,
    PersistentTokenStorage,
)
from zav.agents_sdk.adapters.mcp.strict_schema import ensure_strict_json_schema
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.mcp_oauth import MCP_OAUTH_CONNECT_TOOL_NAME_PREFIX
from zav.agents_sdk.domain.request_headers import RequestHeaders
from zav.agents_sdk.domain.tools import Tool, ToolStreamingConfig


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


class RootPathNormalizingOAuthContext(OAuthContext):
    """PRM `authorization_servers` entries in the wild carry a trailing slash
    on empty-path issuers (https://accounts.google.com/) while the ASM
    `issuer` omits it; the SDK compares the two byte-wise (RFC 8414 §3.3) and
    aborts the whole auth flow on the mismatch. The two forms are RFC 3986
    equivalent, so strip the slash of a bare-root path on every
    `auth_server_url` write (the SDK rewrites the field during 401
    discovery, so normalizing our own writes is not enough)."""

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name == "auth_server_url"
            and isinstance(value, str)
            and value.endswith("/")
            and urlparse(value).path == "/"
        ):
            value = value[:-1]
        super().__setattr__(name, value)


class DeferredOAuthClientProvider(OAuthClientProvider):

    def __init__(
        self,
        *,
        server_url: str,
        client_metadata: OAuthClientMetadata,
        storage: TokenStorage,
        mcp_oauth_store: MCPOAuthIntegrationStore,
        tenant: str,
        server_name: str,
        user_uuid: str,
        pending_auth_urls: Dict[str, str],
        persist_client_info_on_auth: bool = False,
    ):
        super().__init__(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=self.__redirect_handler,
            callback_handler=self.__callback_handler,
        )
        self.context.__class__ = RootPathNormalizingOAuthContext
        self.__store = mcp_oauth_store
        self.__tenant = tenant
        self.__server_name = server_name
        self.__user_uuid = user_uuid
        self.__pending_auth_urls = pending_auth_urls
        self.__persist_client_info_on_auth = persist_client_info_on_auth

    async def __redirect_handler(self, url: str) -> None:
        logger.info(f"OAuth: authorization required for server '{self.__server_name}'")

    async def __callback_handler(self) -> AuthorizationCodeResult:
        raise OAuthFlowError(f"Deferred OAuth for server '{self.__server_name}'")

    async def _initialize(self) -> None:
        await super()._initialize()
        # The MCP SDK only sets `token_expiry_time` after a fresh token response;
        # tokens loaded from persistent storage carry `expires_in` but never get
        # the expiry recorded, so `is_token_valid()` would erroneously return
        # True for already-expired tokens and the SDK would skip the refresh
        # step. Restore the expiry so refresh kicks in when appropriate.
        tokens = self.context.current_tokens
        if tokens and tokens.expires_in is not None:
            self.context.update_token_expiry(tokens)
        # Per MCP spec (§2.3 Authorization Server Discovery), clients MUST use
        # PRM (RFC 9728) and ASM (RFC 8414). The SDK only fetches these lazily
        # after a 401 inside `async_auth_flow`, which means `_refresh_token`
        # would otherwise resolve the token endpoint via the legacy
        # `<server_url>/token` fallback and fail on servers whose token endpoint
        # lives elsewhere.
        # Eagerly discover so refresh and authorization flows always use the
        # endpoints advertised by the authorization server metadata.
        if self.context.oauth_metadata is None:
            await self.__discover_oauth_metadata()

    async def __discover_oauth_metadata(self) -> None:
        async with httpx2.AsyncClient() as client:
            prm_urls = build_protected_resource_metadata_discovery_urls(
                None, self.context.server_url
            )
            for url in prm_urls:
                try:
                    response = await client.send(create_oauth_metadata_request(url))
                except httpx2.HTTPError as exc:
                    logger.warning(
                        f"OAuth: PRM discovery error for server "
                        f"'{self.__server_name}' at {url}: {exc}"
                    )
                    continue
                prm = await handle_protected_resource_response(response)
                if prm:
                    self.context.protected_resource_metadata = prm
                    if prm.authorization_servers:
                        self.context.auth_server_url = str(prm.authorization_servers[0])
                    break
            asm_urls = build_oauth_authorization_server_metadata_discovery_urls(
                self.context.auth_server_url, self.context.server_url
            )
            for url in asm_urls:
                try:
                    response = await client.send(create_oauth_metadata_request(url))
                except httpx2.HTTPError as exc:
                    logger.warning(
                        f"OAuth: ASM discovery error for server "
                        f"'{self.__server_name}' at {url}: {exc}"
                    )
                    continue
                ok, asm = await handle_auth_metadata_response(response)
                if not ok:
                    break
                if asm:
                    self.context.oauth_metadata = asm
                    break

    async def _perform_authorization_code_grant(self) -> tuple[str, str]:
        if self.context.client_metadata.redirect_uris is None:
            raise OAuthFlowError("No redirect URIs provided")
        client_info = self.context.client_info
        if not client_info:
            raise OAuthFlowError("No client info available for authorization")

        # Per MCP spec (§2.3): clients MUST rely on PRM/ASM for endpoint
        # discovery. By the time this runs, `_initialize` has eagerly fetched
        # metadata; if it is still missing, the server is non-compliant.
        if not (
            self.context.oauth_metadata
            and self.context.oauth_metadata.authorization_endpoint
            and self.context.oauth_metadata.token_endpoint
        ):
            raise OAuthFlowError(
                f"OAuth metadata for server '{self.__server_name}' is missing "
                "authorization or token endpoint; cannot start authorization flow."
            )
        auth_endpoint = str(self.context.oauth_metadata.authorization_endpoint)
        token_endpoint = str(self.context.oauth_metadata.token_endpoint)

        pkce_params = PKCEParameters.generate()
        state = secrets.token_urlsafe(32)

        auth_params = {
            "response_type": "code",
            "client_id": client_info.client_id,
            "redirect_uri": str(self.context.client_metadata.redirect_uris[0]),
            "state": state,
            "code_challenge": pkce_params.code_challenge,
            "code_challenge_method": "S256",
        }
        if self.context.should_include_resource_param(self.context.protocol_version):
            auth_params["resource"] = self.context.get_resource_url()
        if self.context.client_metadata.scope:
            auth_params["scope"] = self.context.client_metadata.scope
        # Google-specific extensions (ignored by other providers): without these
        # Google returns only an access_token, no refresh_token, so the SDK has
        # no way to re-authenticate when the access_token expires.
        # `access_type=offline` requests a refresh_token; `prompt=consent`
        # forces a fresh consent so a refresh_token is issued even if the user
        # had previously consented without offline access.
        auth_endpoint_lower = auth_endpoint.lower()
        if (
            "accounts.google.com" in auth_endpoint_lower
            or "oauth2.googleapis.com" in auth_endpoint_lower
        ):
            auth_params["access_type"] = "offline"
            auth_params["prompt"] = "consent"

        authorization_url = f"{auth_endpoint}?{urlencode(auth_params)}"

        pending = MCPOAuthPendingAuthorization(
            state=state,
            tenant=self.__tenant,
            server_name=self.__server_name,
            user_uuid=self.__user_uuid,
            auth_url=authorization_url,
            code_verifier=pkce_params.code_verifier,
            token_endpoint=token_endpoint,
            redirect_uri=str(self.context.client_metadata.redirect_uris[0]),
            resource=(
                self.context.get_resource_url()
                if self.context.should_include_resource_param(
                    self.context.protocol_version
                )
                else None
            ),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
        )
        if self.__persist_client_info_on_auth:
            await self.__store.set_client_info(
                self.__tenant,
                self.__server_name,
                client_info,
            )
        await self.__store.save_pending_authorization(
            tenant=self.__tenant,
            user_uuid=self.__user_uuid,
            server_name=self.__server_name,
            pending=pending,
        )
        self.__pending_auth_urls[self.__server_name] = authorization_url

        logger.info(f"OAuth: authorization required for server '{self.__server_name}'")
        raise OAuthFlowError(
            f"Deferred OAuth for server '{self.__server_name}': "
            "user must authenticate via callback endpoint"
        )


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
    servers: List[MCPServerConfig] = Field(default_factory=list)
    tool_streaming: Optional[Dict[str, ToolStreamingConfig]] = Field(
        default=None,
        description="Optional mapping of tool names to streaming configurations. "
        "If not provided, tools not in this mapping get auto-generated streaming text. "
        "If provided, only tools in this mapping get streamed.",
    )


class _WorkerCall:
    __slots__ = ("fn", "future")

    def __init__(self, fn, future: asyncio.Future):
        self.fn = fn
        self.future = future


class MCPServerWorker:
    """Owns the transport + ClientSession lifecycle for one MCP server inside
    a single dedicated task. All session interactions (initialize, list_tools,
    call_tool) are dispatched as RPCs over a queue so the streamable_http
    transport's anyio TaskGroup never crosses task boundaries. When deferred
    OAuth fires, the TaskGroup teardown stays inside the worker task where
    it's a normal exception instead of a cancel storm leaking into callers.
    """

    def __init__(
        self,
        server_name: str,
        open_session,
        pending_auth_urls: Dict[str, str],
    ):
        self.__server_name = server_name
        self.__open_session = open_session
        self.__pending_auth_urls = pending_auth_urls
        self.__queue: asyncio.Queue[Optional[_WorkerCall]] = asyncio.Queue()
        self.__task: Optional[asyncio.Task] = None
        self.__terminal_error: Optional[BaseException] = None

    async def start(self) -> None:
        loop = asyncio.get_event_loop()
        started: asyncio.Future = loop.create_future()
        self.__task = asyncio.create_task(
            self.__run(started), name=f"mcp-worker[{self.__server_name}]"
        )
        await started

    async def __run(self, started: asyncio.Future) -> None:
        try:
            async with AsyncExitStack() as stack:
                try:
                    session = await self.__open_session(stack)
                except BaseException as e:
                    self.__set_terminal(e)
                    if not started.done():
                        started.set_exception(e)
                    self.__drain_queue()
                    if isinstance(e, asyncio.CancelledError):
                        raise
                    return
                if not started.done():
                    started.set_result(None)
                await self.__serve(session)
        except BaseException as e:
            if self.__terminal_error is None:
                self.__set_terminal(e)
            self.__drain_queue()
            if isinstance(e, asyncio.CancelledError):
                raise

    async def __serve(self, session: ClientSession) -> None:
        while True:
            call = await self.__queue.get()
            if call is None:
                return
            try:
                result = await call.fn(session)
            except OAuthFlowError as e:
                self.__set_terminal(e)
                if not call.future.done():
                    call.future.set_exception(e)
                return
            except asyncio.CancelledError as e:
                # The streamable_http transport's anyio TaskGroup cancels its
                # cancel scope when its inner auth flow raises OAuthFlowError;
                # that surfaces here as CancelledError on our session call.
                # If our DeferredOAuthClientProvider populated pending auth
                # for this server, surface as OAuthFlowError and shut down.
                if self.__pending_auth_urls.get(self.__server_name):
                    err = OAuthFlowError(
                        f"Deferred OAuth for server '{self.__server_name}': "
                        "user must authenticate via callback endpoint"
                    )
                    self.__set_terminal(err)
                    if not call.future.done():
                        call.future.set_exception(err)
                    # Clear our cancel state so the AsyncExitStack can close
                    # without re-raising; the cancel was scope-local, not a
                    # real cancellation of this worker task.
                    current = asyncio.current_task()
                    if current is not None:
                        while current.cancelling() > 0:
                            current.uncancel()
                    return
                if not call.future.done():
                    call.future.set_exception(e)
                raise
            except Exception as e:
                # The transport runs the auth flow in its own background task,
                # so a deferred-OAuth abort can surface on the session call as
                # a closed stream or timeout rather than CancelledError. A
                # pending auth URL populated by our DeferredOAuthClientProvider
                # identifies those: authorization is required.
                if self.__pending_auth_urls.get(self.__server_name):
                    err = OAuthFlowError(
                        f"Deferred OAuth for server '{self.__server_name}': "
                        "user must authenticate via callback endpoint"
                    )
                    self.__set_terminal(err)
                    if not call.future.done():
                        call.future.set_exception(err)
                    return
                if not call.future.done():
                    call.future.set_exception(e)
            else:
                if not call.future.done():
                    call.future.set_result(result)

    def __set_terminal(self, error: BaseException) -> None:
        if self.__terminal_error is None:
            self.__terminal_error = error

    def __drain_queue(self) -> None:
        error = self.__terminal_error or RuntimeError(
            f"MCP worker for '{self.__server_name}' is not running"
        )
        while True:
            try:
                call = self.__queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if call is not None and not call.future.done():
                call.future.set_exception(error)

    async def request(self, fn) -> Any:
        if self.__terminal_error is not None:
            raise self.__terminal_error
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self.__queue.put_nowait(_WorkerCall(fn=fn, future=future))
        return await future

    async def stop(self) -> None:
        if self.__task is None or self.__task.done():
            return
        self.__queue.put_nowait(None)
        try:
            await asyncio.wait_for(asyncio.shield(self.__task), timeout=5.0)
        except (asyncio.TimeoutError, BaseException):
            if not self.__task.done():
                self.__task.cancel()
                try:
                    await self.__task
                except BaseException:
                    pass


class MCPToolsProvider:

    def __init__(
        self,
        mcp_tools_provider_configuration: MCPToolsProviderConfiguration,
        tenant: str,
        mcp_oauth_store: Optional[MCPOAuthIntegrationStore] = None,
        user_uuid: Optional[str] = None,
    ):
        self.__config = mcp_tools_provider_configuration
        self.__mcp_oauth_store = mcp_oauth_store
        self.__tenant = tenant
        self.__user_uuid = user_uuid
        self.__lock = asyncio.Lock()
        self.__workers: Dict[str, MCPServerWorker] = {}
        self.__failed_servers: Dict[str, BaseTransportConfig] = {}
        self.__failed_server_reasons: Dict[str, str] = {}
        self.__connection_states: Dict[str, MCPOAuthConnectionState] = {}
        self.__client_infos: Dict[str, OAuthClientInformationFull] = {}
        self.__pending_auth_urls: Dict[str, str] = {}
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
            workers = list(self.__workers.values())
            # Stop workers concurrently; each worker.stop() is independent and
            # already swallows its own errors, so return_exceptions is belt-and-
            # suspenders against one stop blocking the rest.
            await asyncio.gather(
                *(worker.stop() for worker in workers),
                return_exceptions=True,
            )
            self.__workers = {}

    def __oauth_transport_configs(self) -> Dict[str, BaseTransportConfig]:
        configs: Dict[str, BaseTransportConfig] = {}
        for server_cfg in self.__config.servers:
            transport_cfg = server_cfg.get_transport_config()
            if transport_cfg.oauth_client:
                configs[transport_cfg.name] = transport_cfg
        return configs

    async def __load_oauth_state(self) -> None:
        if not (self.__mcp_oauth_store and self.__tenant and self.__user_uuid):
            return
        oauth_transport_configs = self.__oauth_transport_configs()
        server_names = list(oauth_transport_configs.keys())
        if not server_names:
            return
        self.__connection_states = await self.__mcp_oauth_store.load_user_integrations(
            self.__tenant, self.__user_uuid, server_names
        )
        client_info_server_names = [
            name
            for name, transport_cfg in oauth_transport_configs.items()
            if transport_cfg.oauth_client and not transport_cfg.oauth_client.client_id
        ]
        if client_info_server_names:
            self.__client_infos = await self.__mcp_oauth_store.load_client_info(
                self.__tenant, client_info_server_names
            )

    def __pre_registered_client_info(
        self, oauth_cfg: OauthClientTransportConfig
    ) -> Optional[OAuthClientInformationFull]:
        if not oauth_cfg.client_id:
            return None
        return OAuthClientInformationFull(
            client_id=oauth_cfg.client_id,
            client_secret=oauth_cfg.client_secret,
            **oauth_cfg.client_metadata.model_dump(exclude_none=True),
        )

    def __has_deferred_oauth_dependencies(self) -> bool:
        return bool(self.__mcp_oauth_store and self.__tenant and self.__user_uuid)

    def __missing_deferred_oauth_dependencies(self) -> List[str]:
        missing = []
        if not self.__mcp_oauth_store:
            missing.append("mcp_oauth_store")
        if not self.__tenant:
            missing.append("tenant")
        if not self.__user_uuid:
            missing.append("user_uuid")
        return missing

    async def __connect(self):
        await self.__load_oauth_state()
        # Connect to every server concurrently: each connect is an independent
        # network handshake, so a sequential loop made per-request latency the
        # sum of all handshakes instead of the slowest one. A bare gather()
        # (not return_exceptions) preserves the prior abort-on-genuine-cancel
        # behavior, since each server swallows its own OAuth/connection errors
        # and only re-raises a non-deferred CancelledError.
        await asyncio.gather(
            *(self.__connect_server(server_cfg) for server_cfg in self.__config.servers)
        )

    async def __connect_server(self, server_cfg: "MCPServerConfig") -> None:
        transport_cfg = server_cfg.get_transport_config()
        oauth_cfg = transport_cfg.oauth_client
        state = self.__connection_states.get(transport_cfg.name) if oauth_cfg else None

        if oauth_cfg:
            if state and state.pending and not state.tokens:
                self.__failed_servers[transport_cfg.name] = transport_cfg
                self.__failed_server_reasons[transport_cfg.name] = (
                    "authorization is already pending"
                )
                return
            if not self.__has_deferred_oauth_dependencies():
                missing = self.__missing_deferred_oauth_dependencies()
                logger.warning(
                    f"Skipping OAuth MCP server '{transport_cfg.name}': "
                    "deferred OAuth dependencies are not available: "
                    f"{', '.join(missing)}."
                )
                self.__failed_servers[transport_cfg.name] = transport_cfg
                self.__failed_server_reasons[transport_cfg.name] = (
                    "missing deferred OAuth dependencies: " f"{', '.join(missing)}"
                )
                return

        open_session = self.__make_open_session(server_cfg, transport_cfg, state)
        worker = MCPServerWorker(
            transport_cfg.name, open_session, self.__pending_auth_urls
        )
        try:
            await worker.start()
        except OAuthFlowError as e:
            await worker.stop()
            if (
                state
                and state.tokens
                and self.__mcp_oauth_store
                and self.__tenant
                and self.__user_uuid
            ):
                await self.__mcp_oauth_store.clear_tokens(
                    self.__tenant, self.__user_uuid, transport_cfg.name
                )
            logger.warning(
                f"Failed to authorize MCP server '{transport_cfg.name}': {e}. "
                "Skipping; placeholder tools will be registered."
            )
            self.__failed_servers[transport_cfg.name] = transport_cfg
            self.__failed_server_reasons[transport_cfg.name] = str(e)
        except asyncio.CancelledError:
            await worker.stop()
            if transport_cfg.name not in self.__pending_auth_urls:
                raise
            logger.warning(
                f"Failed to authorize MCP server '{transport_cfg.name}': "
                "deferred OAuth requires user authorization. Skipping; "
                "placeholder tools will be registered."
            )
            self.__failed_servers[transport_cfg.name] = transport_cfg
            self.__failed_server_reasons[transport_cfg.name] = (
                "deferred OAuth requires user authorization"
            )
        except Exception as e:
            await worker.stop()
            logger.warning(
                f"Failed to connect to MCP server '{transport_cfg.name}': {e}. "
                "Skipping; placeholder tools will be registered."
            )
            self.__failed_servers[transport_cfg.name] = transport_cfg
            self.__failed_server_reasons[transport_cfg.name] = "connection error"
        else:
            self.__workers[transport_cfg.name] = worker

    def __make_open_session(
        self,
        server_cfg: "MCPServerConfig",
        transport_cfg: BaseTransportConfig,
        state: Optional[MCPOAuthConnectionState],
    ):
        async def open_session(stack: AsyncExitStack) -> ClientSession:
            oauth_auth = None
            if oauth_cfg := transport_cfg.oauth_client:
                # OAuth servers without these dependencies are filtered out in
                # `__connect` before we get here; re-check to narrow Optional.
                mcp_oauth_store = self.__mcp_oauth_store
                user_uuid = self.__user_uuid
                if mcp_oauth_store is None or user_uuid is None:
                    raise RuntimeError(
                        f"OAuth server '{transport_cfg.name}' is missing "
                        "deferred OAuth dependencies; should have been "
                        "filtered in __connect."
                    )
                client_info = self.__client_infos.get(transport_cfg.name)
                pre_registered_client_info = self.__pre_registered_client_info(
                    oauth_cfg
                )
                if pre_registered_client_info:
                    client_info = pre_registered_client_info

                storage: TokenStorage = PersistentTokenStorage(
                    mcp_oauth_store,
                    self.__tenant,
                    user_uuid,
                    transport_cfg.name,
                    tokens=state.tokens if state else None,
                    client_info=client_info,
                )
                client_metadata = OAuthClientMetadata.model_validate(
                    oauth_cfg.client_metadata
                )
                oauth_auth = DeferredOAuthClientProvider(
                    server_url=oauth_cfg.server_url,
                    client_metadata=client_metadata,
                    storage=storage,
                    mcp_oauth_store=mcp_oauth_store,
                    tenant=self.__tenant,
                    server_name=transport_cfg.name,
                    user_uuid=user_uuid,
                    pending_auth_urls=self.__pending_auth_urls,
                    persist_client_info_on_auth=(
                        pre_registered_client_info is not None
                    ),
                )
            if isinstance(transport_cfg, StdIoTransportConfig):
                transport = await stack.enter_async_context(
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
                transport = await stack.enter_async_context(
                    sse_client(
                        url=transport_cfg.url,
                        headers=transport_cfg.headers,
                        timeout=transport_cfg.timeout,
                        sse_read_timeout=transport_cfg.sse_read_timeout,
                        auth=oauth_auth,
                    )
                )
            elif isinstance(transport_cfg, StreamableHttpTransportConfig):
                # streamable_http_client no longer accepts headers/timeouts/auth;
                # they are carried by a caller-owned httpx2 client. Enter it on
                # the stack before the transport so it outlives the transport.
                http_client = await stack.enter_async_context(
                    httpx2.AsyncClient(
                        headers=transport_cfg.headers,
                        timeout=httpx2.Timeout(
                            transport_cfg.timeout.total_seconds(),
                            read=transport_cfg.sse_read_timeout.total_seconds(),
                        ),
                        auth=oauth_auth,
                        follow_redirects=True,
                    )
                )
                transport = await stack.enter_async_context(
                    streamable_http_client(
                        url=transport_cfg.url,
                        http_client=http_client,
                        terminate_on_close=transport_cfg.terminate_on_close,
                    )
                )
            else:
                raise ValueError(f"Unsupported transport: {server_cfg.transport}")
            read, write, *_ = transport
            session = await stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                    read_timeout_seconds=(
                        self.__config.client_session_timeout_seconds
                        if self.__config.client_session_timeout_seconds
                        else None
                    ),
                )
            )
            await session.initialize()
            return session

        return open_session

    async def get_tools(self) -> List[Tool]:
        """
        Discover MCP tools from all configured servers and
        return them as Tool instances.
        """
        if not self.__config.enabled:
            return []
        resolved = await self.__resolve_server_tools()
        return [t for tools in resolved.values() for t in tools]

    def __failed_server_auth_url(self, server_name: str) -> Optional[str]:
        auth_url = self.__pending_auth_urls.get(server_name)
        if auth_url:
            return auth_url
        state = self.__connection_states.get(server_name)
        if state and state.pending:
            return state.pending.auth_url
        return None

    def __authorization_required_response(
        self, server_name: str, failure_reason: str
    ) -> Dict[str, Any]:
        if self.__failed_server_auth_url(server_name):
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
                auth_url = self.__failed_server_auth_url(server_name)
                if auth_url:
                    visible["authorization_url"] = auth_url
                    visible["action_label"] = f"Authorize {server_name}"
                return visible
            return response

        return transform

    def __failed_server_tools(
        self, server_name: str, failure_reason: str, auth_url: Optional[str]
    ) -> List[Tool]:
        async def placeholder(**kwargs: Any) -> Dict[str, str]:
            if auth_url:
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

        def transform(response: Optional[Dict]) -> Optional[Dict]:
            if response is None:
                return None
            visible_response: Dict[str, Any] = {
                "server_name": server_name,
                "status": response.get("status"),
                "display_text": response.get("display_text"),
            }
            if response.get("status") == "authorization_required" and auth_url:
                visible_response["authorization_url"] = auth_url
                visible_response["action_label"] = f"Authorize {server_name}"
            if response.get("reason"):
                visible_response["reason"] = response["reason"]
            return visible_response

        return [
            Tool(
                name=f"{MCP_OAUTH_CONNECT_TOOL_NAME_PREFIX}{server_name}",
                description=(
                    f"Call this when the user asks to connect, use, or check the "
                    f"status of {server_name}. Returns the connection status and "
                    "whether user authorization is required."
                ),
                executable=placeholder,
                parameters_spec={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                streaming_config=ToolStreamingConfig(
                    running_text=f"Checking {server_name} connection...",
                    completed_text="{{ display_text }}",
                    response_transform=transform,
                ),
            ),
        ]

    async def __resolve_server_tools(
        self,
    ) -> Dict[str, List[Tool]]:
        if self.__resolved is not None:
            return self.__resolved
        if not self.__workers and not self.__failed_servers:
            await self.__connect()
        self.__resolved = {}
        # List tools from every connected server concurrently: each list_tools is
        # an independent RPC, so the sequential loop made setup latency the sum of
        # all of them (on top of the connect handshakes).
        await asyncio.gather(
            *(
                self.__list_server_tools(server_cfg, self.__resolved)
                for server_cfg in self.__config.servers
            )
        )

        for server_name in self.__failed_servers:
            auth_url = self.__failed_server_auth_url(server_name)
            failure_reason = self.__failed_server_reasons.get(
                server_name, "unknown MCP connection failure"
            )
            self.__resolved[server_name] = self.__failed_server_tools(
                server_name, failure_reason, auth_url
            )

        return self.__resolved

    async def __list_server_tools(
        self, server_cfg: "MCPServerConfig", resolved: Dict[str, List[Tool]]
    ) -> None:
        transport_cfg = server_cfg.get_transport_config()
        server_name = transport_cfg.name
        worker = self.__workers.get(server_name)
        if worker is None:
            return
        try:
            mcp_tools = await worker.request(lambda s: s.list_tools())
        except Exception as e:
            logger.warning(
                f"Failed to list tools for MCP server '{server_name}': {e}. "
                "Skipping; placeholder tools will be registered."
            )
            self.__failed_servers[server_name] = transport_cfg
            self.__failed_server_reasons[server_name] = str(e)
            await worker.stop()
            self.__workers.pop(server_name, None)
            return
        server_tools: List[Tool] = []
        for mcp_tool in mcp_tools.tools:
            schema = dict(mcp_tool.input_schema or {})
            if "properties" not in schema:
                schema["properties"] = {}
            if self.__config.convert_schemas_to_strict:
                try:
                    schema = ensure_strict_json_schema(schema)
                except Exception as e:
                    logger.info(f"Error converting MCP schema to strict mode: {e}")

            invoke_fn = self.__make_invoke(worker, mcp_tool, server_name)

            streaming_config: ToolStreamingConfig | None = None
            if self.__config.tool_streaming is None:
                fallback = f"Completed {mcp_tool.name}"
                streaming_config = ToolStreamingConfig(
                    running_text=f"Running {mcp_tool.name}...",
                    completed_text="{{ display_text or '" + fallback + "' }}",
                    response_transform=self.__make_auth_aware_transform(server_name),
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
        resolved[server_name] = server_tools

    def __make_invoke(self, worker: MCPServerWorker, tool: MCPTool, server_name: str):
        tool_name = tool.name

        tool_call_timeout = self.__config.tool_call_timeout_seconds

        async def invoke(**kwargs: Any) -> Any:
            try:
                result = await worker.request(
                    lambda s: s.call_tool(
                        tool_name, kwargs, read_timeout_seconds=tool_call_timeout
                    )
                )
            except OAuthFlowError as e:
                return self.__authorization_required_response(server_name, str(e))
            except Exception as e:
                if self.__failed_server_auth_url(server_name):
                    return self.__authorization_required_response(server_name, str(e))
                raise
            items = result.content or []
            if len(items) == 1:
                return items[0].model_dump_json()
            return json.dumps(
                {"parts": [item.model_dump(mode="json") for item in items]}
            )

        return invoke


class MCPToolsProviderFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        request_headers: RequestHeaders,
        mcp_oauth_store: MCPOAuthIntegrationStore,
        tenant: str = "zetaalpha",
        mcp_tools_provider_configuration: MCPToolsProviderConfiguration = (
            MCPToolsProviderConfiguration()
        ),
    ) -> MCPToolsProvider:
        return MCPToolsProvider(
            mcp_tools_provider_configuration,
            mcp_oauth_store=mcp_oauth_store,
            tenant=tenant,
            user_uuid=request_headers.requester_uuid,
        )
