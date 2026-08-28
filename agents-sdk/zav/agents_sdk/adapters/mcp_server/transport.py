import contextvars
import json
from dataclasses import dataclass
from typing import Any, Optional

import mcp.types as mcp_types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.exceptions import MCPError
from starlette.responses import Response
from zav.logging import logger
from zav.message_bus import MessageBus
from zav.pydantic_compat import pydantic_encoder

from zav.agents_sdk.domain.mcp_server_setup import MCPServerSetupNotFound
from zav.agents_sdk.domain.request_headers import RequestHeaders
from zav.agents_sdk.handlers import commands


@dataclass
class MCPRequestState:
    """What the transport carries from the authenticated route into the
    protocol handlers: the identity the dependencies resolved, the addressed
    setup, and the bus the handlers dispatch on."""

    tenant: str
    request_headers: RequestHeaders
    mcp_server_identifier: Optional[str]
    index_id: Optional[str]
    llm_configuration_name: Optional[str]
    message_bus: MessageBus


_request_state: contextvars.ContextVar[Optional[MCPRequestState]] = (
    contextvars.ContextVar("mcp_request_state", default=None)
)


def _current_state() -> MCPRequestState:
    state = _request_state.get()
    if state is None:
        raise MCPError(
            code=mcp_types.INTERNAL_ERROR,
            message="MCP request state is not bound.",
        )
    return state


async def _on_list_tools(
    ctx: Any, params: Optional[mcp_types.PaginatedRequestParams]
) -> mcp_types.ListToolsResult:
    state = _current_state()
    try:
        results = await state.message_bus.handle(
            commands.ListMCPTools(
                tenant=state.tenant,
                request_headers=state.request_headers,
                mcp_server_identifier=state.mcp_server_identifier,
                index_id=state.index_id,
                llm_configuration_name=state.llm_configuration_name,
            )
        )
        tools = results.pop(0)
    except MCPServerSetupNotFound as exc:
        raise MCPError(code=mcp_types.INVALID_PARAMS, message=str(exc))
    except MCPError:
        raise
    except Exception:
        logger.exception("MCP list_tools failed")
        raise MCPError(
            code=mcp_types.INTERNAL_ERROR, message="Failed to list MCP tools."
        )
    return mcp_types.ListToolsResult(
        tools=[
            mcp_types.Tool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.get_parameters_spec(),
            )
            for tool in tools
        ]
    )


async def _on_call_tool(
    ctx: Any, params: mcp_types.CallToolRequestParams
) -> mcp_types.CallToolResult:
    state = _current_state()
    try:
        results = await state.message_bus.handle(
            commands.CallMCPTool(
                tenant=state.tenant,
                request_headers=state.request_headers,
                mcp_server_identifier=state.mcp_server_identifier,
                index_id=state.index_id,
                llm_configuration_name=state.llm_configuration_name,
                tool_name=params.name,
                arguments=dict(params.arguments or {}),
            )
        )
        result = results.pop(0)
        text = (
            result
            if isinstance(result, str)
            else json.dumps(result, default=pydantic_encoder)
        )
    except MCPServerSetupNotFound as exc:
        raise MCPError(code=mcp_types.INVALID_PARAMS, message=str(exc))
    except MCPError:
        raise
    except Exception as e:
        logger.exception("MCP call_tool %s failed", params.name)
        return mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(
                    type="text", text=f"Tool '{params.name}' failed: {e}"
                )
            ],
            is_error=True,
        )
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)]
    )


def build_mcp_server() -> Server:
    """The protocol engine: the mcp low-level Server wired to dispatch
    ListMCPTools / CallMCPTool on the message bus. This module is the whole
    mcp-library surface: routing, identity, and authorization live on the
    FastAPI route; tool resolution and execution live in the command handlers.
    Tool failures carry the exception message, mirroring what the agent loop
    feeds the model when a tool raises."""
    server: Server = Server("zeta-alpha-knowledge")
    server.add_request_handler(
        "tools/list", mcp_types.PaginatedRequestParams, _on_list_tools
    )
    server.add_request_handler(
        "tools/call", mcp_types.CallToolRequestParams, _on_call_tool
    )
    return server


class MCPTransport(Response):
    """Bridges a FastAPI route to the mcp protocol engine. The route runs the
    standard dependency functions, builds the request state, and returns this
    response; Starlette then invokes it with the raw ASGI connection. In
    stateless mode the session manager keeps nothing between requests, so each
    request runs its own and no app lifespan is needed."""

    def __init__(self, state: MCPRequestState):
        super().__init__()
        self.__state = state

    async def __call__(self, scope, receive, send) -> None:
        # One session manager per request instead of one in an app lifespan: in
        # stateless mode the manager keeps nothing between requests, so this is
        # equivalent, and its cost (a task group + a Server with two handler
        # registrations) is noise next to any tool call.
        session_manager = StreamableHTTPSessionManager(
            app=build_mcp_server(), stateless=True, json_response=True
        )
        token = _request_state.set(self.__state)
        try:
            async with session_manager.run():
                await session_manager.handle_request(scope, receive, send)
        finally:
            _request_state.reset(token)
