from typing import Callable, List, Optional, Tuple

from fastapi import APIRouter, FastAPI

from zav.api.utils import add_exception_handlers, use_route_names_as_operation_ids


def setup_routers(
    app: FastAPI,
    routers: List[Tuple[str, APIRouter]],
    extra_exception_handlers: Optional[List[Callable[[FastAPI], None]]] = None,
):

    add_exception_handlers(app)
    if extra_exception_handlers:
        for handler in extra_exception_handlers:
            handler(app)
    for prefix, router in routers:
        if isinstance(router, APIRouter):
            app.include_router(router, prefix=prefix)

    use_route_names_as_operation_ids(app)
