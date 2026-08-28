from zav.agents_sdk.adapters.mcp_server.tool_surface import (
    ToolSurface,
    ToolSurfaceFactory,
)

__all__ = ["ToolSurface", "ToolSurfaceFactory"]

try:
    # The transport pulls in the mcp server library, an optional extra. Hosts
    # that never serve MCP can run without it; the tool surface stays importable.
    from zav.agents_sdk.adapters.mcp_server.transport import (
        MCPRequestState,
        MCPTransport,
    )

    __all__ += ["MCPRequestState", "MCPTransport"]
except ImportError:
    pass
