from typing import Any, Dict, List, Optional

from mcp.types import Tool as MCPTool

from zav.agents_sdk.domain.tools import ALL_CALLERS, ToolCaller

# The MCP Apps extension (specification/2026-01-26/apps.mdx in
# github.com/modelcontextprotocol/ext-apps): a tool names the `ui://` resource
# a host renders its result with, and who may call it.
APP_VIEWS_EXTENSION_ID = "io.modelcontextprotocol/ui"
APP_VIEW_MIME_TYPE = "text/html;profile=mcp-app"
# Off until the sandbox proxy has its own origin, as the spec requires: no
# configuration renders an MCP server's view (docs/decisions/0005).
MCP_APP_VIEWS = False
# The extension's earlier flat key, which hosts still read.
_LEGACY_RESOURCE_URI_KEY = "ui/resourceUri"


def app_views_capability() -> Dict[str, Dict[str, Any]]:
    """What a client that renders views advertises at initialize."""
    return {APP_VIEWS_EXTENSION_ID: {"mimeTypes": [APP_VIEW_MIME_TYPE]}}


def _ui_meta(mcp_tool: MCPTool) -> Dict[str, Any]:
    ui = (mcp_tool.meta or {}).get("ui")
    return ui if isinstance(ui, dict) else {}


def tool_view(mcp_tool: MCPTool) -> Optional[str]:
    uri = _ui_meta(mcp_tool).get("resourceUri") or (mcp_tool.meta or {}).get(
        _LEGACY_RESOURCE_URI_KEY
    )
    return uri if isinstance(uri, str) and uri.startswith("ui://") else None


def tool_callers(mcp_tool: MCPTool) -> List[ToolCaller]:
    """The extension's `visibility`: "model" is the model and its REPL, "app"
    the views. Unlike the spec's default, a view calls only the tools that
    name "app", as for our own tools."""
    visibility = _ui_meta(mcp_tool).get("visibility")
    if not isinstance(visibility, list):
        visibility = ["model"]
    callers: List[ToolCaller] = list(ALL_CALLERS) if "model" in visibility else []
    if "app" in visibility:
        callers.append("app")
    return callers


def result_for_model(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What the model reads of a tool result kept whole for its view: the
    content, not the structured result. Anything else the provider answers
    with (a connection needing authorization) passes as is."""
    if result is None or "content" not in result:
        return result
    return {key: result[key] for key in ("content", "isError") if key in result}
