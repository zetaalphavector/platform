import asyncio
import logging
from typing import Callable, Dict, List, Optional, Tuple, Type

from fastapi import APIRouter, FastAPI
from zav.logging import get_logger
from zav.message_bus import Bootstrap, Command, MessageBus

from zav.api.dependencies import get_message_bus
from zav.api.probes import CommandHandlerRegistry as ProbesCommandHandlerRegistry
from zav.api.probes import probes_router
from zav.api.setup_routers import setup_routers


class MessageBusDependency:
    def __init__(self, bootstrap: Bootstrap):

        self.__input_bootstrap = bootstrap
        self.__bootstrap: Optional[Bootstrap] = None
        self.__message_bus: Optional[MessageBus] = None
        self.__lock = asyncio.Lock()

    async def __call__(self):

        if self.__message_bus is not None:
            return self.__message_bus

        async with self.__lock:
            if self.__message_bus is not None:
                return self.__message_bus
            if self.__bootstrap is not None:
                self.__message_bus = self.__bootstrap.message_bus
                return self.__message_bus
            self.__bootstrap = self.__input_bootstrap
            await self.__bootstrap.startup()
            self.__message_bus = self.__bootstrap.message_bus

        return self.__message_bus

    async def close(self):

        if self.__bootstrap is not None:
            await self.__bootstrap.shutdown()


def setup_api(
    app: FastAPI,
    bootstrap: Bootstrap,
    routers: List[Tuple[str, APIRouter]],
    extra_exception_handlers: Optional[List[Callable[[FastAPI], None]]] = None,
    extra_command_handler_registry: Optional[Dict[Type[Command], Callable]] = None,
):

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
    message_bus_dep = MessageBusDependency(bootstrap)

    @app.on_event("shutdown")
    async def shutdown():
        await message_bus_dep.close()

    # Add probes router
    app.include_router(probes_router)

    setup_routers(
        app=app, routers=routers, extra_exception_handlers=extra_exception_handlers
    )

    app.dependency_overrides[get_message_bus] = message_bus_dep
