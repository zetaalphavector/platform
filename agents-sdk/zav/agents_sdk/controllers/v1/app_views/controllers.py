from typing import Optional

from fastapi import APIRouter, Depends, Query
from zav.api.dependencies import get_message_bus
from zav.api.errors import ForbiddenException, NotFoundException
from zav.message_bus import MessageBus

from zav.agents_sdk.adapters.app_views import AppViewNotFound
from zav.agents_sdk.controllers.v1.app_views.types import (
    AppToolCallForm,
    AppToolCallItem,
    AppViewForm,
    AppViewItem,
)
from zav.agents_sdk.controllers.v1.common import get_headers
from zav.agents_sdk.domain import RequestHeaders
from zav.agents_sdk.domain.tools import ToolCallRefused
from zav.agents_sdk.handlers import commands

app_views_router = APIRouter(tags=["chat"])


@app_views_router.post(
    "/chat/app-views",
    response_model=AppViewItem,
    name="read_app_view",
    operation_id="read_app_view",
)
async def read_app_view(
    body: AppViewForm,
    tenant: str = Query(...),
    index_id: Optional[str] = Query(None),
    request_headers: RequestHeaders = Depends(get_headers),
    message_bus: MessageBus = Depends(get_message_bus),
):
    """The interactive view a tool step renders with, read from the tool's MCP
    server or tools source. A view that ships with the client is not served
    here."""
    try:
        results = await message_bus.handle(
            commands.ReadAppView(
                tenant=tenant,
                request_headers=request_headers,
                index_id=index_id,
                **body.dict(),
            )
        )
    except AppViewNotFound as e:
        raise NotFoundException(str(e))
    view = results.pop(0)
    if view is None:
        raise NotFoundException(
            f"The view of '{body.view_tool}' ships with the client."
        )
    return AppViewItem(html=view.html, ui_meta=view.ui_meta)


@app_views_router.post(
    "/chat/app-views/tool-calls",
    response_model=AppToolCallItem,
    name="call_app_tool",
    operation_id="call_app_tool",
)
async def call_app_tool(
    body: AppToolCallForm,
    tenant: str = Query(...),
    index_id: Optional[str] = Query(None),
    request_headers: RequestHeaders = Depends(get_headers),
    message_bus: MessageBus = Depends(get_message_bus),
):
    """Runs a call a view makes, without a model turn. The tool rules apply;
    a call that would ask the user first is refused."""
    try:
        results = await message_bus.handle(
            commands.CallAppTool(
                tenant=tenant,
                request_headers=request_headers,
                index_id=index_id,
                **body.dict(),
            )
        )
    except AppViewNotFound as e:
        raise NotFoundException(str(e))
    except ToolCallRefused as e:
        raise ForbiddenException(str(e))
    return AppToolCallItem(result=results.pop(0))
