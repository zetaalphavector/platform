import inspect
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Type,
)

from zav.message_bus.common import Command, Event
from zav.message_bus.errors import ExceptionHandlerRegistry
from zav.message_bus.handler_registry import (
    CommandHandlerRegistry,
    EventHandlerRegistry,
    StreamCommandHandlerRegistry,
)
from zav.message_bus.message_bus import MessageBus


@dataclass
class BootstrapDependency:

    name: str
    value: Optional[Any] = None
    context_value: Optional[Callable[[], AsyncContextManager]] = None
    startup_fn: Optional[Callable[[], Awaitable[None]]] = None
    shutdown_fn: Optional[Callable[[], Awaitable[None]]] = None


def inject_dependencies(
    handler: Callable, dependencies: List[BootstrapDependency]
) -> Callable:

    params = inspect.signature(handler).parameters

    value_deps = {
        dependency.name: dependency.value
        for dependency in dependencies
        if dependency.name in params and dependency.value is not None
    }
    context_deps = {
        dependency.name: dependency.context_value
        for dependency in dependencies
        if dependency.name in params and dependency.context_value is not None
    }

    async def handler_wrap(*args, **kwargs):
        if context_deps != {}:
            async with AsyncExitStack() as stack:
                resolved_value_deps = {
                    name: await stack.enter_async_context(ctxmanager())
                    for name, ctxmanager in context_deps.items()
                }
                return await handler(
                    *args, **kwargs, **value_deps, **resolved_value_deps
                )

        return await handler(*args, **kwargs, **value_deps)

    return handler_wrap


def inject_dependencies_streaming(
    handler: Callable, dependencies: List["BootstrapDependency"]
) -> Callable[..., AsyncGenerator[Any, None]]:
    """Inject dependencies into a streaming (async-generator) handler.

    Unlike ``inject_dependencies`` which opens an ``AsyncExitStack`` around a
    single ``await``, this wrapper opens the stack inside the returned async
    generator and holds it open for the full lifetime of the iteration. This
    means context-managed dependencies (``BootstrapDependency.context_value``)
    survive across every ``yield`` and are cleaned up only when the consumer
    stops iterating, either by exhaustion, exception, or ``aclose()``.
    """

    params = inspect.signature(handler).parameters

    value_deps = {
        dependency.name: dependency.value
        for dependency in dependencies
        if dependency.name in params and dependency.value is not None
    }
    context_deps = {
        dependency.name: dependency.context_value
        for dependency in dependencies
        if dependency.name in params and dependency.context_value is not None
    }

    async def streaming_handler_wrap(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        async with AsyncExitStack() as stack:
            resolved_value_deps = {
                name: await stack.enter_async_context(ctxmanager())
                for name, ctxmanager in context_deps.items()
            }
            inner = handler(*args, **kwargs, **value_deps, **resolved_value_deps)
            try:
                async for item in inner:
                    yield item
            finally:
                # Close the inner handler generator promptly on early consumer
                # exit (e.g. client disconnect) so its own finally/async with
                # cleanup runs deterministically instead of waiting for GC.
                await inner.aclose()

    return streaming_handler_wrap


class Bootstrap:
    def __init__(
        self,
        dependencies: List[BootstrapDependency],
        command_handler_registry: Type[CommandHandlerRegistry],
        event_handler_registry: Type[EventHandlerRegistry],
        exception_handler_registry: Optional[Type[ExceptionHandlerRegistry]] = None,
        stream_command_handler_registry: Optional[
            Type[StreamCommandHandlerRegistry]
        ] = None,
    ):

        self.__dependencies = dependencies
        self.__command_handler_registry = command_handler_registry.registry
        self.__event_handler_registry = event_handler_registry.registry
        self.__exception_handler_registry = exception_handler_registry
        self.__stream_command_handler_registry = (
            stream_command_handler_registry.registry
            if stream_command_handler_registry
            else {}
        )

    async def startup(self):

        for dep in self.__dependencies:
            if dep.startup_fn is not None:
                await dep.startup_fn()

    async def shutdown(self):

        for dep in self.__dependencies:
            if dep.shutdown_fn is not None:
                await dep.shutdown_fn()

    def update_command_handler_registry(self, update: Dict[Type[Command], Callable]):

        self.__command_handler_registry = {
            **self.__command_handler_registry,
            **update,
        }

    def update_event_handler_registry(self, update: Dict[Type[Event], List[Callable]]):

        self.__event_handler_registry = {
            **self.__event_handler_registry,
            **{
                event: [*handlers, *self.__event_handler_registry.get(event, [])]
                for event, handlers in update.items()
            },
        }

    def update_stream_command_handler_registry(
        self, update: Dict[Type[Command], Callable]
    ):

        self.__stream_command_handler_registry = {
            **self.__stream_command_handler_registry,
            **update,
        }

    @property
    def message_bus(self):

        injected_stream_command_handlers = {
            command_type: inject_dependencies_streaming(handler, self.__dependencies)
            for command_type, handler in self.__stream_command_handler_registry.items()
        }
        injected_command_handlers = {
            command_type: inject_dependencies(handler, self.__dependencies)
            for command_type, handler in self.__command_handler_registry.items()
        }
        injected_event_handlers = {
            event_type: [
                inject_dependencies(handler, self.__dependencies)
                for handler in handlers
            ]
            for event_type, handlers in self.__event_handler_registry.items()
        }
        return MessageBus(
            command_handlers=injected_command_handlers,
            event_handlers=injected_event_handlers,
            stream_command_handlers=injected_stream_command_handlers,
            exception_handlers=(
                self.__exception_handler_registry.registry
                if self.__exception_handler_registry
                else {}
            ),
        )
