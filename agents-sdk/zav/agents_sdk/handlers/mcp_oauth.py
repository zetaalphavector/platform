from typing import List

from zav.logging import logger
from zav.message_bus import CommandHandlerRegistry, HandlerException, Message

from zav.agents_sdk.adapters.mcp.oauth_token_client import (
    MCPOAuthTokenClient,
    MCPOAuthTokenExchangeError,
)
from zav.agents_sdk.adapters.mcp.oauth_token_storage import MCPOAuthIntegrationStore
from zav.agents_sdk.domain.mcp_oauth import MCPOAuthCallbackResult
from zav.agents_sdk.handlers import commands


@CommandHandlerRegistry.register(commands.HandleMCPOAuthCallback)
async def handle_mcp_oauth_callback(
    cmd: commands.HandleMCPOAuthCallback,
    queue: List[Message],
    mcp_oauth_store: MCPOAuthIntegrationStore,
    mcp_oauth_token_client: MCPOAuthTokenClient,
):
    pending = await mcp_oauth_store.get_pending_authorization(cmd.state)
    if not pending:
        raise HandlerException("Invalid or expired OAuth state")

    if cmd.error:
        await mcp_oauth_store.clear_pending_authorization(cmd.state)
        return MCPOAuthCallbackResult(status="error")

    if not cmd.code:
        raise HandlerException("Missing OAuth authorization code")

    client_infos = await mcp_oauth_store.load_client_info(
        pending.tenant, [pending.server_name]
    )
    client_info = client_infos.get(pending.server_name)
    if not client_info:
        await mcp_oauth_store.clear_pending_authorization(cmd.state)
        raise HandlerException("Missing OAuth client info")

    try:
        tokens = await mcp_oauth_token_client.exchange_authorization_code(
            pending, client_info, cmd.code
        )
    except MCPOAuthTokenExchangeError as exception:
        logger.error(
            f"Token exchange failed for server '{pending.server_name}': {exception}"
        )
        await mcp_oauth_store.clear_pending_authorization(cmd.state)
        raise HandlerException("Token exchange failed")

    await mcp_oauth_store.save_tokens(
        pending.tenant,
        pending.user_uuid,
        pending.server_name,
        tokens,
    )
    return MCPOAuthCallbackResult(status="success")
