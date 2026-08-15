from dataclasses import dataclass
from typing import Any, Dict, Type, Union

from zav.pydantic_compat import BaseModel


class ErrorResponse(BaseModel):
    detail: str


@dataclass(frozen=True)
class ApiError:
    """One HTTP error a route can return. Define it once, then list the ones a
    route surfaces via ``responses=error_responses(ERROR_A, ERROR_B)``.

    ``model`` is the response body schema; it defaults to the shared
    ``ErrorResponse`` (``{"detail": str}``). Errors whose body carries extra
    fields supply their own ``BaseModel`` instead — e.g. a 409 whose ``detail``
    is a structured object."""

    status_code: int
    description: str
    model: Type[BaseModel] = ErrorResponse


def error_responses(*errors: ApiError) -> Dict[Union[int, str], Dict[str, Any]]:
    """Build a FastAPI ``responses=`` mapping for the given errors. Each body
    schema lands in ``components/schemas`` once and is ``$ref``-ed from every
    operation that uses it."""
    return {
        error.status_code: {"model": error.model, "description": error.description}
        for error in errors
    }


ERROR_BAD_REQUEST = ApiError(400, "Bad request.")
ERROR_FORBIDDEN = ApiError(403, "Forbidden.")
ERROR_NOT_FOUND = ApiError(404, "Not found.")
ERROR_INTERNAL = ApiError(500, "Internal server error.")

# The errors the global exception handlers can raise on any route. Applied once
# when routers are mounted (see ``setup_routers``), so no controller repeats them.
COMMON_ERROR_RESPONSES = error_responses(
    ERROR_BAD_REQUEST, ERROR_FORBIDDEN, ERROR_NOT_FOUND, ERROR_INTERNAL
)
