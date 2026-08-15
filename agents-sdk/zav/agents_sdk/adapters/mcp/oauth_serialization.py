from datetime import datetime, timedelta, timezone
from typing import Optional

from mcp.shared.auth import OAuthToken


def oauth_token_from_storage(
    *,
    access_token: Optional[str],
    refresh_token: Optional[str],
    token_type: Optional[str],
    scope: Optional[str],
    expires_at: Optional[datetime],
) -> Optional[OAuthToken]:
    if not access_token or not token_type:
        return None
    expires_in: Optional[int] = None
    if expires_at is not None:
        remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
        expires_in = max(int(remaining), 0)
    return OAuthToken.model_validate(
        {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": token_type,
            "expires_in": expires_in,
            "scope": scope,
        }
    )


def oauth_token_expires_at(tokens: OAuthToken) -> Optional[datetime]:
    if tokens.expires_in is None:
        return None
    return datetime.now(timezone.utc) + timedelta(seconds=tokens.expires_in)
