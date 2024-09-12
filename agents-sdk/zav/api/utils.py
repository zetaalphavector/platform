from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError
from zav.message_bus import HandlerException

from zav.api.errors import (
    BadRequestException,
    ForbiddenException,
    NotFoundException,
    UnknownException,
)


def add_exception_handlers(app: FastAPI):
    @app.exception_handler(BadRequestException)
    async def bad_request_handler(request: Request, exc: BadRequestException):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(NotFoundException)
    async def not_found_handler(request: Request, exc: NotFoundException):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ForbiddenException)
    async def forbidden_handler(request: Request, exc: ForbiddenException):
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(HandlerException)
    async def handler_exception_handler(request: Request, exc: HandlerException):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(UnknownException)
    async def unknown_handler(request: Request, exc: UnknownException):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": str(exc)})


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    """Simplify operation IDs.

    Should be called only after all routes have been added.
    """
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name
