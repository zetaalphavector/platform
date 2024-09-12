from typing import Optional

from fastapi import Header

from zav.agents_sdk.domain import RequestHeaders


def get_headers(
    requester_uuid: Optional[str] = Header(None, alias="requester-uuid"),
    user_roles: Optional[str] = Header(None, alias="user-roles"),
    user_tenants: Optional[str] = Header(None, alias="user-tenants"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_auth: Optional[str] = Header(None, alias="X-Auth"),
):
    return RequestHeaders(
        **{
            "requester_uuid": requester_uuid,
            "user_roles": user_roles,
            "user_tenants": user_tenants,
            "authorization": authorization,
            "x_auth": x_auth,
        }
    )
