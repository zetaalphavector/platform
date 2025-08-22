from typing import Optional

from zav.pydantic_compat import PYDANTIC_V2, BaseModel, ConfigDict, Field


class RequestHeaders(BaseModel):
    requester_uuid: Optional[str] = Field(None, alias="requester-uuid")
    user_roles: Optional[str] = Field(None, alias="user-roles")
    user_tenants: Optional[str] = Field(None, alias="user-tenants")
    authorization: Optional[str] = Field(None, alias="Authorization")
    x_auth: Optional[str] = Field(None, alias="X-Auth")

    if PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    else:

        class Config:
            orm_mode = True
            allow_population_by_field_name = True

    def __bool__(self) -> bool:
        # check if the user has set any of the fields to consider the object truthy
        return bool(self.dict(exclude_unset=True))
