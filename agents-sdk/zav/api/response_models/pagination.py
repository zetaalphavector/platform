from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel
from pydantic.version import VERSION as PYDANTIC_VERSION

PYDANTIC_V2 = PYDANTIC_VERSION.startswith("2.")

if PYDANTIC_V2:
    from pydantic import BaseModel as GenericModel
    from pydantic import model_validator

    validator = model_validator(mode="after")
else:
    from pydantic.class_validators import root_validator
    from pydantic.generics import GenericModel  # type: ignore

    validator = root_validator()  # type: ignore

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

    @validator
    @classmethod
    def check_consistency(cls, values):
        # For Pydantic v1, treat 'values' as a dictionary.
        # For Pydantic v2, treat 'values' as an instance of the model.
        if PYDANTIC_V2:
            # Access model attributes
            total = values.count
            page_size = values.page_size
            page = values.page
        else:
            # Access dictionary keys
            total = values["count"]
            page_size = values["page_size"]
            page = values["page"]

        remaining = total - page_size * page
        next_params = (
            PageParams(page=page + 1, page_size=page_size) if remaining > 0 else None
        )
        previous_params = (
            PageParams(page=page - 1, page_size=page_size) if page > 1 else None
        )

        # Assign next and previous parameters based on Pydantic version
        if PYDANTIC_V2:
            if next_params:
                values.next = next_params
            elif "next" in values.model_fields_set:
                del values.next
            if previous_params:
                values.previous = previous_params
            elif "previous" in values.model_fields_set:
                del values.previous
        else:
            if next_params:
                values["next"] = next_params
            else:
                values.pop("next", None)
            if previous_params:
                values["previous"] = previous_params
            else:
                values.pop("previous", None)

        return values
