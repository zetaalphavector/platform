from typing import Literal

from zav.pydantic_compat import BaseModel

# Prefix for the synthetic placeholder tool registered when an MCP server
# requires user OAuth authorization. The tool name is
# `f"{MCP_OAUTH_CONNECT_TOOL_NAME_PREFIX}{server_name}"`. Frontends use the
# prefix as the single, stable hook to render an "Authorize" UI for any MCP
# server.
MCP_OAUTH_CONNECT_TOOL_NAME_PREFIX: str = "mcp_oauth_connect_"


class MCPOAuthCallbackResult(BaseModel):
    status: Literal["success", "error"]
