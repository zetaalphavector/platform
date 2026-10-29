import asyncio
import secrets
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Set, cast
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
    get_client_metadata_scopes,
    handle_auth_metadata_response,
    handle_protected_resource_response,
)
from mcp.client.session import IncomingMessage, MessageHandlerFnT
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from mcp.types import CallToolResult
from mcp.types import Tool as MCPTool
from mcp.types import ToolListChangedNotification
from zav.logging import logger

from zav.agents_sdk.adapters.mcp.oauth_repository import (
    MCPOAuthPendingAuthorization,
    MCPToolListing,
)
from zav.agents_sdk.adapters.mcp.oauth_token_storage import (
    MCPOAuthConnectionState,
    MCPOAuthIntegrationStore,
    PersistentTokenStorage,
)
from zav.agents_sdk.adapters.mcp.server_config import (
    BaseTransportConfig,
    MCPServerConfig,
    OauthClientTransportConfig,
    SseTransportConfig,
    StdIoTransportConfig,
    StreamableHttpTransportConfig,
    server_fingerprint,
)


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

    pinned_auth_server_url: Optional[str] = None

    def __setattr__(self, name: str, value: Any) -> None:
        # A configured authorization server wins over the one the resource
        # advertises: a cluster-internal MCP server names itself, which the
        # user's browser cannot reach.
        if name == "auth_server_url" and self.pinned_auth_server_url:
            value = self.pinned_auth_server_url
        if (
            name == "auth_server_url"
            and isinstance(value, str)
            and value.endswith("/")
            and urlparse(value).path == "/"
        ):
            value = value[:-1]
        super().__setattr__(name, value)

    def should_include_resource_param(
        self, protocol_version: Optional[str] = None
    ) -> bool:
        # Entra v2 rejects `resource` next to `api://.../scope` scopes.
        if self.pinned_auth_server_url:
            return False
        return super().should_include_resource_param(protocol_version)


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
        authorization_server_url: Optional[str] = None,
    ):
        super().__init__(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=self.__redirect_handler,
            callback_handler=self.__callback_handler,
        )
        self.context.__class__ = RootPathNormalizingOAuthContext
        if authorization_server_url:
            cast(
                RootPathNormalizingOAuthContext, self.context
            ).pinned_auth_server_url = authorization_server_url
            self.context.auth_server_url = authorization_server_url
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

    async def start_authorization(self) -> None:
        """Asks for the user's grant before the server refuses a request:
        some servers (Google's) list their tools to anyone and only refuse a
        tool call. Raises OAuthFlowError once the authorization is pending."""
        async with self.context.lock:
            await self._initialize()
            self.context.client_metadata.scope = (
                get_client_metadata_scopes(
                    None,
                    self.context.protected_resource_metadata,
                    self.context.oauth_metadata,
                    self.context.client_metadata.grant_types,
                )
                or self.context.client_metadata.scope
            )
            await self._perform_authorization_code_grant()

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
        self.__interrupted_call: Optional[_WorkerCall] = None

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
                    if self.__is_transport_cancellation(e):
                        raise
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
            self.__set_terminal(self.__failure_cause(e))
            error = self.__terminal_error or e
            for future in (
                started,
                self.__interrupted_call.future if self.__interrupted_call else None,
            ):
                if future is not None and not future.done():
                    future.set_exception(error)
            self.__drain_queue()
            if isinstance(e, asyncio.CancelledError):
                raise

    def __is_transport_cancellation(self, error: BaseException) -> bool:
        # A transport TaskGroup cancels its host when a background task fails
        # (e.g. the OAuth flow raising OAuthFlowError on a resource mismatch).
        # The real cause only surfaces once the stack unwinds, so the caller's
        # future is resolved there instead of with this scope-local cancel.
        return isinstance(error, asyncio.CancelledError) and not (
            self.__pending_auth_urls.get(self.__server_name)
        )

    def __failure_cause(self, error: BaseException) -> BaseException:
        if isinstance(error, BaseExceptionGroup):
            for inner in error.exceptions:
                cause = self.__failure_cause(inner)
                if not isinstance(cause, asyncio.CancelledError):
                    return cause
        return error

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
            except asyncio.CancelledError:
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
                self.__interrupted_call = call
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


class MCPServerUnavailableError(Exception):
    pass


MCPConnectionStatus = Literal[
    "not_connected", "needs_authorization", "connected", "failed", "disabled"
]


@dataclass(frozen=True)
class MCPConnectResult:
    server_name: str
    status: Literal["connected", "needs_authorization", "failed"]
    tools: List[MCPTool] = field(default_factory=list)
    authorization_url: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class MCPServerConnection:
    """The user's connection to one server. `tools` is the server's listing
    while it is current for the server's configuration."""

    server_name: str
    status: MCPConnectionStatus
    tools: List[MCPTool] = field(default_factory=list)
    listed_at: Optional[datetime] = None
    error: Optional[str] = None
    failed_at: Optional[datetime] = None
    authorization_url: Optional[str] = None
    # The user's choice per tool (the server's own tool name).
    tool_permissions: Dict[str, str] = field(default_factory=dict)


SessionOpener = Callable[[MCPServerConfig, AsyncExitStack], Awaitable[ClientSession]]


# A server and its new tools, after its open session reported a change.
ListingListener = Callable[[MCPServerConfig, List[MCPTool]], Awaitable[None]]


class MCPConnectionService:
    """One user's connections to MCP servers: connecting them (OAuth
    included), their status and tool listings, and calling their tools.

    With a connection store, a server is dialled on connect and on its first
    tool call, and its listing is served from the store in between. Without
    one (a bare CLI run, a test) every server is dialled when asked for its
    status."""

    def __init__(
        self,
        tenant: str,
        user_uuid: Optional[str] = None,
        connection_store: Optional[MCPOAuthIntegrationStore] = None,
        client_session_timeout_seconds: float = 2,
        tool_call_timeout_seconds: float = 30,
        session_opener: Optional[SessionOpener] = None,
    ):
        self.__tenant = tenant
        self.__user_uuid = user_uuid
        self.__store = connection_store if user_uuid else None
        self.__client_session_timeout_seconds = client_session_timeout_seconds
        self.__tool_call_timeout_seconds = tool_call_timeout_seconds
        self.__session_opener = session_opener
        self.__lock = asyncio.Lock()
        self.__dial_locks: Dict[str, asyncio.Lock] = {}
        self.__workers: Dict[str, MCPServerWorker] = {}
        self.__failure_reasons: Dict[str, str] = {}
        self.__pending_auth_urls: Dict[str, str] = {}
        self.__states: Dict[str, MCPOAuthConnectionState] = {}
        self.__client_infos: Dict[str, OAuthClientInformationFull] = {}
        self.__listing_listeners: List[ListingListener] = []
        self.__refreshes: Set["asyncio.Task[None]"] = set()

    def on_listing_changed(self, listener: "ListingListener") -> None:
        """Called with a server and its new tools when an open session reports
        its tool list changed; the new listing is stored first."""
        self.__listing_listeners.append(listener)

    async def connections(
        self, servers: List[MCPServerConfig]
    ) -> Dict[str, MCPServerConnection]:
        if self.__store is None:
            results = await asyncio.gather(
                *(self.__dial(server_cfg) for server_cfg in servers)
            )
            return {
                result.server_name: MCPServerConnection(
                    server_name=result.server_name,
                    status=result.status,
                    tools=result.tools,
                    error=result.error,
                    authorization_url=result.authorization_url,
                )
                for result in results
            }
        await self.__load_states(servers)
        return {
            server_cfg.get_transport_config().name: self.__stored_connection(server_cfg)
            for server_cfg in servers
        }

    async def connect(
        self, server_cfg: MCPServerConfig, reauthorize: bool = False
    ) -> MCPConnectResult:
        """Dials the server and lists its tools, storing them as the user's
        connection. A server that needs the user's authorization answers with
        the URL to authorize it; `reauthorize` asks for a new grant even when
        the user has one."""
        server_name = server_cfg.get_transport_config().name
        if reauthorize:
            await self.__drop_tokens(server_cfg)
        await self.__load_states([server_cfg])
        await self.__drop_tokens_from_other_sign_in(server_cfg)
        self.__failure_reasons.pop(server_name, None)
        result = await self.__dial(server_cfg)
        if result.status == "connected" and self.__lacks_grant(server_cfg):
            result = await self.__start_authorization(server_cfg)
        if result.status == "failed":
            await self.__record_failure(server_name, result.error or "")
        return result

    async def call(
        self, server_cfg: MCPServerConfig, tool_name: str, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Calls one of the server's tools, dialling the server on its first
        call. Raises MCPServerUnavailableError when the server cannot be
        reached or needs the user's authorization."""
        server_name = server_cfg.get_transport_config().name
        worker = await self.__worker_for(server_cfg)
        try:
            return await worker.request(
                lambda s: s.call_tool(
                    tool_name,
                    arguments,
                    read_timeout_seconds=self.__tool_call_timeout_seconds,
                )
            )
        except OAuthFlowError as e:
            await self.__stop_worker(server_name)
            raise MCPServerUnavailableError(str(e)) from e
        except Exception as e:
            if self.authorization_url(server_name):
                await self.__stop_worker(server_name)
                raise MCPServerUnavailableError(str(e)) from e
            raise

    def authorization_url(self, server_name: str) -> Optional[str]:
        auth_url = self.__pending_auth_urls.get(server_name)
        if auth_url:
            return auth_url
        state = self.__states.get(server_name)
        if state and state.pending:
            return state.pending.auth_url
        return None

    async def cleanup(self) -> None:
        async with self.__lock:
            workers = list(self.__workers.values())
            await asyncio.gather(
                *(worker.stop() for worker in workers),
                return_exceptions=True,
            )
            self.__workers = {}

    def __stored_connection(self, server_cfg: MCPServerConfig) -> MCPServerConnection:
        server_name = server_cfg.get_transport_config().name
        state = self.__states.get(server_name)
        listing = state.tool_listing if state else None
        current = listing is not None and listing.fingerprint == server_fingerprint(
            server_cfg
        )
        status: MCPConnectionStatus
        if state and state.disabled:
            status = "disabled"
        elif state and state.pending:
            status = "needs_authorization"
        elif current and not self.__lacks_grant(server_cfg):
            status = "connected"
        elif state and state.error:
            status = "failed"
        else:
            status = "not_connected"
        return MCPServerConnection(
            server_name=server_name,
            status=status,
            tools=(
                [MCPTool.model_validate(t) for t in listing.tools]
                if listing and current
                else []
            ),
            listed_at=listing.listed_at if listing and current else None,
            error=state.error if state else None,
            failed_at=state.failed_at if state else None,
            authorization_url=(
                state.pending.auth_url
                if state and state.pending and status == "needs_authorization"
                else None
            ),
            tool_permissions=dict(state.tool_permissions) if state else {},
        )

    async def __dial(self, server_cfg: MCPServerConfig) -> MCPConnectResult:
        server_name = server_cfg.get_transport_config().name
        async with self.__dial_locks.setdefault(server_name, asyncio.Lock()):
            if server_name not in self.__workers:
                await self.__connect_server(server_cfg)
            mcp_tools = await self.__list_tools(server_cfg)
        if mcp_tools is not None:
            return MCPConnectResult(
                server_name=server_name, status="connected", tools=mcp_tools
            )
        auth_url = self.authorization_url(server_name)
        if auth_url:
            return MCPConnectResult(
                server_name=server_name,
                status="needs_authorization",
                authorization_url=auth_url,
            )
        return MCPConnectResult(
            server_name=server_name,
            status="failed",
            error=self.__failure_reasons.get(server_name, "connection error"),
        )

    async def __worker_for(self, server_cfg: MCPServerConfig) -> MCPServerWorker:
        """The server's session, opened on its first tool call. It lists the
        tools again, which keeps the stored list current at no cost: the MCP
        client lists them before its first call anyway."""
        server_name = server_cfg.get_transport_config().name
        async with self.__dial_locks.setdefault(server_name, asyncio.Lock()):
            worker = self.__workers.get(server_name)
            if worker is not None:
                return worker
            # The user may have authorized the server since the turn read its
            # state, from the card an earlier call of this turn showed.
            await self.__load_states([server_cfg])
            await self.__connect_server(server_cfg)
            if server_name not in self.__workers or (
                await self.__list_tools(server_cfg) is None
            ):
                reason = self.__failure_reasons.get(server_name, "connection error")
                if not self.authorization_url(server_name):
                    await self.__record_failure(server_name, reason)
                raise MCPServerUnavailableError(reason)
            return self.__workers[server_name]

    async def __load_states(self, servers: List[MCPServerConfig]) -> None:
        if self.__store is None or not servers:
            return
        names = [server_cfg.get_transport_config().name for server_cfg in servers]
        states = await self.__store.load_user_integrations(
            self.__tenant, cast(str, self.__user_uuid), names
        )
        for name in names:
            self.__states.pop(name, None)
            if name in states:
                self.__states[name] = states[name]
        client_info_server_names = [
            server_cfg.get_transport_config().name
            for server_cfg in servers
            if (oauth_cfg := server_cfg.get_transport_config().oauth_client)
            and not oauth_cfg.client_id
        ]
        if client_info_server_names:
            self.__client_infos.update(
                await self.__store.load_client_info(
                    self.__tenant, client_info_server_names
                )
            )

    async def __drop_tokens_from_other_sign_in(
        self, server_cfg: MCPServerConfig
    ) -> None:
        server_name = server_cfg.get_transport_config().name
        state = self.__states.get(server_name)
        listing = state.tool_listing if state else None
        if (
            self.__store is None
            or state is None
            or state.tokens is None
            or listing is None
            or listing.fingerprint == server_fingerprint(server_cfg)
        ):
            return
        await self.__store.clear_tokens(
            self.__tenant, cast(str, self.__user_uuid), server_name
        )
        await self.__load_states([server_cfg])

    def __lacks_grant(self, server_cfg: MCPServerConfig) -> bool:
        """An OAuth server the user has not authorized: its tools act as the
        user, so a listing it served without a grant does not connect it."""
        transport_cfg = server_cfg.get_transport_config()
        state = self.__states.get(transport_cfg.name)
        return (
            transport_cfg.oauth_client is not None
            and self.__session_opener is None
            and not (state and state.tokens)
        )

    async def __drop_tokens(self, server_cfg: MCPServerConfig) -> None:
        server_name = server_cfg.get_transport_config().name
        await self.__stop_worker(server_name)
        if self.__store is not None:
            await self.__store.clear_tokens(
                self.__tenant, cast(str, self.__user_uuid), server_name
            )

    async def __stop_worker(self, server_name: str) -> None:
        """Drops a session that cannot serve again (its authorization
        failed), so the next call dials with the user's current grant."""
        worker = self.__workers.pop(server_name, None)
        if worker is not None:
            await worker.stop()

    async def __start_authorization(
        self, server_cfg: MCPServerConfig
    ) -> MCPConnectResult:
        transport_cfg = server_cfg.get_transport_config()
        server_name = transport_cfg.name
        await self.__stop_worker(server_name)
        provider = self.__oauth_provider(transport_cfg, self.__states.get(server_name))
        try:
            await cast(DeferredOAuthClientProvider, provider).start_authorization()
        except OAuthFlowError as e:
            self.__failure_reasons[server_name] = str(e)
        auth_url = self.authorization_url(server_name)
        if auth_url:
            return MCPConnectResult(
                server_name=server_name,
                status="needs_authorization",
                authorization_url=auth_url,
            )
        return MCPConnectResult(
            server_name=server_name,
            status="failed",
            error=self.__failure_reasons.get(server_name, "authorization error"),
        )

    async def __record_failure(self, server_name: str, error: str) -> None:
        if self.__store is not None:
            await self.__store.record_failure(
                self.__tenant, cast(str, self.__user_uuid), server_name, error
            )

    async def __connect_server(self, server_cfg: MCPServerConfig) -> None:
        transport_cfg = server_cfg.get_transport_config()
        oauth_cfg = transport_cfg.oauth_client
        state = self.__states.get(transport_cfg.name) if oauth_cfg else None
        # A new dial: an authorization URL an earlier one got may be spent.
        self.__pending_auth_urls.pop(transport_cfg.name, None)

        if oauth_cfg and self.__session_opener is None:
            if state and state.pending:
                self.__failure_reasons[transport_cfg.name] = (
                    "authorization is already pending"
                )
                return
            if self.__store is None:
                logger.warning(
                    f"Skipping OAuth MCP server '{transport_cfg.name}': "
                    "no user connection store to keep its tokens."
                )
                self.__failure_reasons[transport_cfg.name] = (
                    "no user connection store to keep its tokens"
                )
                return

        worker = MCPServerWorker(
            transport_cfg.name,
            self.__make_open_session(server_cfg, transport_cfg, state),
            self.__pending_auth_urls,
        )
        try:
            await worker.start()
        except OAuthFlowError as e:
            await worker.stop()
            if state and state.tokens and self.__store is not None:
                await self.__store.clear_tokens(
                    self.__tenant, cast(str, self.__user_uuid), transport_cfg.name
                )
            logger.warning(
                f"Failed to authorize MCP server '{transport_cfg.name}': {e}."
            )
            self.__failure_reasons[transport_cfg.name] = str(e)
        except asyncio.CancelledError:
            await worker.stop()
            if transport_cfg.name not in self.__pending_auth_urls:
                raise
            logger.warning(
                f"Failed to authorize MCP server '{transport_cfg.name}': "
                "deferred OAuth requires user authorization."
            )
            self.__failure_reasons[transport_cfg.name] = (
                "deferred OAuth requires user authorization"
            )
        except Exception as e:
            await worker.stop()
            logger.warning(
                f"Failed to connect to MCP server '{transport_cfg.name}': {e}."
            )
            self.__failure_reasons[transport_cfg.name] = "connection error"
        else:
            self.__workers[transport_cfg.name] = worker

    async def __list_tools(
        self, server_cfg: MCPServerConfig
    ) -> Optional[List[MCPTool]]:
        server_name = server_cfg.get_transport_config().name
        worker = self.__workers.get(server_name)
        if worker is None:
            return None
        try:
            mcp_tools = (await worker.request(lambda s: s.list_tools())).tools
        except Exception as e:
            logger.warning(f"Failed to list tools for MCP server '{server_name}': {e}.")
            self.__failure_reasons[server_name] = str(e)
            await worker.stop()
            self.__workers.pop(server_name, None)
            return None
        if self.__store is not None:
            await self.__store.save_tool_listing(
                self.__tenant,
                cast(str, self.__user_uuid),
                server_name,
                MCPToolListing(
                    fingerprint=server_fingerprint(server_cfg),
                    listed_at=datetime.now(timezone.utc),
                    tools=[
                        t.model_dump(mode="json", by_alias=True, exclude_none=True)
                        for t in mcp_tools
                    ],
                ),
            )
        return list(mcp_tools)

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

    def __oauth_provider(
        self,
        transport_cfg: BaseTransportConfig,
        state: Optional[MCPOAuthConnectionState],
    ) -> Optional[DeferredOAuthClientProvider]:
        oauth_cfg = transport_cfg.oauth_client
        if oauth_cfg is None:
            return None
        # `__connect_server` skips OAuth servers without a store.
        store = cast(MCPOAuthIntegrationStore, self.__store)
        user_uuid = cast(str, self.__user_uuid)
        client_info = self.__client_infos.get(transport_cfg.name)
        pre_registered_client_info = self.__pre_registered_client_info(oauth_cfg)
        if pre_registered_client_info:
            client_info = pre_registered_client_info

        storage: TokenStorage = PersistentTokenStorage(
            store,
            self.__tenant,
            user_uuid,
            transport_cfg.name,
            tokens=state.tokens if state else None,
            client_info=client_info,
        )
        return DeferredOAuthClientProvider(
            server_url=oauth_cfg.server_url,
            client_metadata=OAuthClientMetadata.model_validate(
                oauth_cfg.client_metadata
            ),
            storage=storage,
            mcp_oauth_store=store,
            tenant=self.__tenant,
            server_name=transport_cfg.name,
            user_uuid=user_uuid,
            pending_auth_urls=self.__pending_auth_urls,
            persist_client_info_on_auth=pre_registered_client_info is not None,
            authorization_server_url=oauth_cfg.authorization_server_url,
        )

    def __make_open_session(
        self,
        server_cfg: MCPServerConfig,
        transport_cfg: BaseTransportConfig,
        state: Optional[MCPOAuthConnectionState],
    ) -> Callable[[AsyncExitStack], Awaitable[ClientSession]]:
        session_opener = self.__session_opener
        if session_opener is not None:
            return lambda stack: session_opener(server_cfg, stack)

        async def open_session(stack: AsyncExitStack) -> ClientSession:
            oauth_auth = self.__oauth_provider(transport_cfg, state)
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
                    message_handler=self.__listing_watcher(server_cfg),
                    read_timeout_seconds=(
                        self.__client_session_timeout_seconds
                        if self.__client_session_timeout_seconds
                        else None
                    ),
                )
            )
            await session.initialize()
            return session

        return open_session

    def __listing_watcher(self, server_cfg: MCPServerConfig) -> MessageHandlerFnT:
        async def on_message(message: IncomingMessage) -> None:
            if isinstance(message, ToolListChangedNotification):
                # Listing goes through the server's worker, so it runs as its
                # own task rather than inside the session's receive loop.
                task = asyncio.create_task(self.__refresh_listing(server_cfg))
                self.__refreshes.add(task)
                task.add_done_callback(self.__refreshes.discard)

        return on_message

    async def __refresh_listing(self, server_cfg: MCPServerConfig) -> None:
        mcp_tools = await self.__list_tools(server_cfg)
        if mcp_tools is None:
            return
        for listener in self.__listing_listeners:
            await listener(server_cfg, mcp_tools)
