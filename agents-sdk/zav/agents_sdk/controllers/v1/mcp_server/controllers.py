from typing import Optional

from fastapi import APIRouter, Depends, Query
from zav.api.dependencies import get_message_bus
from zav.message_bus import MessageBus

from zav.agents_sdk.adapters.mcp_server.transport import (
    MCPRequestState,
    MCPTransport,
)
from zav.agents_sdk.controllers.v1.common import get_headers
from zav.agents_sdk.domain import RequestHeaders

# The MCP endpoint as a regular FastAPI route, so the standard dependency
# functions (identity headers, message bus, and any host-added guards such as
# tenant membership) run in front of MCP traffic like on every other endpoint.
# Mounting is a host decision (`za agents serve` stays agents-only), so this
# router is not collected into the default routers: hosts include it in their
# router composition, wrapped with their own guards.
mcp_server_router = APIRouter(prefix="/mcp")


# Kept out of the OpenAPI schema: the endpoint speaks MCP, not the service's
# REST contract, so it must not churn generated clients.
@mcp_server_router.api_route(
    "/{tenant}/",
    methods=["POST", "GET", "DELETE"],
    include_in_schema=False,
)
async def mcp_transport(
    tenant: str,
    mcp_server_identifier: Optional[str] = Query(None),
    index_id: Optional[str] = Query(None),
    llm_configuration_name: Optional[str] = Query(None),
    request_headers: RequestHeaders = Depends(get_headers),
    message_bus: MessageBus = Depends(get_message_bus),
) -> MCPTransport:
    return MCPTransport(
        MCPRequestState(
            tenant=tenant,
            request_headers=request_headers,
            mcp_server_identifier=mcp_server_identifier,
            index_id=index_id,
            llm_configuration_name=llm_configuration_name,
            message_bus=message_bus,
        )
    )
