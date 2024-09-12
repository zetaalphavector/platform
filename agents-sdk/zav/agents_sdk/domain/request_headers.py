from typing import Optional

from pydantic import BaseModel, Field


class RequestHeaders(BaseModel):
    requester_uuid: Optional[str] = Field(None, alias="requester-uuid")
    user_roles: Optional[str] = Field(None, alias="user-roles")
    user_tenants: Optional[str] = Field(None, alias="user-tenants")
    authorization: Optional[str] = Field(None, alias="Authorization")
    x_auth: Optional[str] = Field(None, alias="X-Auth")

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
