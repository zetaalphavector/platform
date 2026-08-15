import base64
from typing import Dict, Tuple
from urllib.parse import quote

import httpx
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from zav.agents_sdk.adapters.mcp.oauth_repository import (
    MCPOAuthPendingAuthorization,
)


class MCPOAuthTokenExchangeError(Exception):
    pass


class MCPOAuthTokenClient:
    async def exchange_authorization_code(
        self,
        pending: MCPOAuthPendingAuthorization,
        client_info: OAuthClientInformationFull,
        code: str,
    ) -> OAuthToken:
        if not client_info.client_id:
            raise MCPOAuthTokenExchangeError("Missing client_id on OAuth client info")
        client_id = client_info.client_id
        token_data: Dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": pending.redirect_uri,
            "client_id": client_id,
            "code_verifier": pending.code_verifier,
        }
        if pending.resource:
            token_data["resource"] = pending.resource

        # RFC 6749 §5.1 mandates a JSON token response, but some providers
        # default to form-urlencoded and only switch to JSON when the client
        # asks for it explicitly.
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        token_data, headers = self.__prepare_token_auth(
            token_data, headers, client_info
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    pending.token_endpoint,
                    data=token_data,
                    headers=headers,
                )
        except httpx.HTTPError as exception:
            raise MCPOAuthTokenExchangeError(
                "Token exchange request failed"
            ) from exception

        if response.status_code != 200:
            raise MCPOAuthTokenExchangeError(
                f"Token exchange failed with status {response.status_code}"
            )

        try:
            token = OAuthToken.model_validate_json(response.content)
            return token
        except Exception as exception:
            raise MCPOAuthTokenExchangeError(
                "Token exchange returned an invalid token response"
            ) from exception

    def __prepare_token_auth(
        self,
        token_data: Dict[str, str],
        headers: Dict[str, str],
        client_info: OAuthClientInformationFull,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        if (
            client_info.token_endpoint_auth_method == "client_secret_basic"
            and client_info.client_secret
            and client_info.client_id
        ):
            encoded_id = quote(client_info.client_id, safe="")
            encoded_secret = quote(client_info.client_secret, safe="")
            credentials = f"{encoded_id}:{encoded_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_credentials}"
        elif (
            client_info.token_endpoint_auth_method == "client_secret_post"
            and client_info.client_secret
        ):
            token_data["client_secret"] = client_info.client_secret
        return token_data, headers


def get_mcp_oauth_token_client() -> MCPOAuthTokenClient:
    return MCPOAuthTokenClient()
