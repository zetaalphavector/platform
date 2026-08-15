from zav.agents_sdk.handlers.chats import *

try:
    # The MCP OAuth callback handler depends on the optional ``mcp`` extra. When
    # it is not installed, simply skip registering the handler instead of
    # crashing the whole bootstrap/handlers import chain.
    from zav.agents_sdk.handlers.mcp_oauth import *
except ImportError:
    pass
