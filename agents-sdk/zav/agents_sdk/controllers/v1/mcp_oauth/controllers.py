from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from zav.api.dependencies import get_message_bus
from zav.api.errors import BadRequestException, UnknownException
from zav.message_bus import HandlerException, MessageBus

from zav.agents_sdk.handlers import commands

mcp_oauth_router = APIRouter(tags=["mcp-oauth"])
setattr(mcp_oauth_router, "public", True)

CALLBACK_SUCCESS_HTML = """<!DOCTYPE html>
<html>
<head><title>Authorization Complete</title></head>
<body>
<h1>Authorization complete.</h1>
<p>You can close this window.</p>
</body>
</html>"""

CALLBACK_ERROR_HTML = """<!DOCTYPE html>
<html>
<head><title>Authorization Failed</title></head>
<body>
<h1>Authorization failed.</h1>
<p>You can close this window and try connecting again.</p>
</body>
</html>"""


def extract_mcp_oauth_callback_command(
    state: str = Query(...),
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
) -> commands.HandleMCPOAuthCallback:
    return commands.HandleMCPOAuthCallback(state=state, code=code, error=error)


@mcp_oauth_router.get(
    "/mcp/oauth/callback",
    response_class=HTMLResponse,
    name="mcp_oauth_callback",
    operation_id="mcp_oauth_callback",
)
async def mcp_oauth_callback(
    command=Depends(extract_mcp_oauth_callback_command),
    message_bus: MessageBus = Depends(get_message_bus),
):
    try:
        results = await message_bus.handle(command)
        result = results.pop(0)
    except HandlerException as exception:
        raise BadRequestException(str(exception))
    except Exception as exception:
        raise UnknownException(f"Could not complete MCP OAuth callback: {exception}")

    if result.status == "error":
        return HTMLResponse(content=CALLBACK_ERROR_HTML)
    return HTMLResponse(content=CALLBACK_SUCCESS_HTML)
