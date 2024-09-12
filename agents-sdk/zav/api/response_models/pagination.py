from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel
from pydantic.class_validators import root_validator
from pydantic.generics import GenericModel

T = TypeVar("T")


class PageParams(BaseModel):

    page: int
    page_size: int


class PaginatedResponse(GenericModel, Generic[T]):

    count: int
    results: List[T]
    next: Optional[PageParams] = None
    previous: Optional[PageParams] = None
    page: int = 1
    page_size: int = 10

    @root_validator()
    @classmethod
    def check_consistency(cls, values):

        total = values["count"]
        page_size = values["page_size"]
        page = values["page"]
        remaining = total - page_size * page
        if remaining > 0:
            values["next"] = PageParams(page=page + 1, page_size=page_size)
        if page > 1:
            values["previous"] = PageParams(page=page - 1, page_size=page_size)

        if values["next"] is None:
            del values["next"]
        if values["previous"] is None:
            del values["previous"]

        return values
