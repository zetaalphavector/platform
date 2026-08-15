import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Dict, List, Mapping, Optional, Tuple, Type

from fastapi import APIRouter, FastAPI
from zav.logging import get_logger
from zav.message_bus import Bootstrap, Command

from zav.api.app_resources import ApiLifespan, compose_lifespans
from zav.api.dependencies import get_message_bus
from zav.api.probes import CommandHandlerRegistry as ProbesCommandHandlerRegistry
from zav.api.probes import probes_router
from zav.api.setup_routers import setup_routers


def __message_bus_lifespan(bootstrap: Bootstrap) -> ApiLifespan:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[Mapping[str, object]]:
        await bootstrap.startup()
        try:
            yield get_message_bus.bind(bootstrap.message_bus)
        finally:
            await bootstrap.shutdown()

    return lifespan


def setup_api(
    bootstrap: Bootstrap,
    routers: List[Tuple[str, APIRouter]],
    app: Optional[FastAPI] = None,
    extra_exception_handlers: Optional[List[Callable[[FastAPI], None]]] = None,
    extra_command_handler_registry: Optional[Dict[Type[Command], Callable]] = None,
    lifespans: Optional[List[ApiLifespan]] = None,
) -> FastAPI:

    if app is not None:
        raise ValueError("setup_api owns the FastAPI lifespan; call it without app")

    logging.getLogger("uvicorn").handlers = []
    logging.getLogger("uvicorn.error").handlers = []
    logging.getLogger("fastapi").handlers = []
    uvicorn_logger = logging.getLogger("uvicorn.access")
    get_logger(uvicorn_logger)

    # Update handler registry with probes handler
    if extra_command_handler_registry is not None:
        bootstrap.update_command_handler_registry(
            {
                **ProbesCommandHandlerRegistry.registry,
                **extra_command_handler_registry,
            }
        )
    else:
        bootstrap.update_command_handler_registry(ProbesCommandHandlerRegistry.registry)

    app = FastAPI(
        lifespan=compose_lifespans(
            [__message_bus_lifespan(bootstrap), *(lifespans or [])]
        )
    )

    app.include_router(probes_router)

    setup_routers(
        app=app, routers=routers, extra_exception_handlers=extra_exception_handlers
    )

    return app
