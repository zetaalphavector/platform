from typing import Callable, Dict, List, Optional, Type

from zav.message_bus.common import Command, Event


class CommandHandlerRegistry:

    registry: Dict[Type[Command], Callable] = {}

    @classmethod
    def register(cls, command: Type[Command]) -> Callable:
        def inner_wrapper(wrapped_function: Callable) -> Callable:
            cls.registry[command] = wrapped_function
            return wrapped_function

        return inner_wrapper

    @classmethod
    def merge(cls, other_registry: Dict[Type[Command], Callable]) -> None:
        """Merge handlers from another registry.

        Only registers if the command is not already registered,
        or if the existing handler has a different name.
        """
        for command, handler in other_registry.items():
            existing_handler: Optional[Callable] = cls.registry.get(command)
            if existing_handler is None:
                cls.registry[command] = handler
            elif getattr(existing_handler, "__name__", None) != getattr(
                handler, "__name__", None
            ):
                cls.registry[command] = handler


class StreamCommandHandlerRegistry:
    """Registry for streaming command handlers.

    A streaming handler is an ``async def`` that returns an ``AsyncGenerator``.
    It is invoked via ``MessageBus.handle_stream`` rather than ``handle``, and
    its dependency ``AsyncExitStack`` is held open for the lifetime of the
    generator iteration (see ``inject_dependencies_streaming``).
    """

    registry: Dict[Type[Command], Callable] = {}

    @classmethod
    def register(cls, command: Type[Command]) -> Callable:
        def inner_wrapper(wrapped_function: Callable) -> Callable:
            cls.registry[command] = wrapped_function
            return wrapped_function

        return inner_wrapper

    @classmethod
    def merge(cls, other_registry: Dict[Type[Command], Callable]) -> None:
        for command, handler in other_registry.items():
            existing_handler: Optional[Callable] = cls.registry.get(command)
            if existing_handler is None:
                cls.registry[command] = handler
            elif getattr(existing_handler, "__name__", None) != getattr(
                handler, "__name__", None
            ):
                cls.registry[command] = handler


class EventHandlerRegistry:

    registry: Dict[Type[Event], List[Callable]] = {}

    @classmethod
    def register(cls, event: Type[Event]) -> Callable:
        def inner_wrapper(wrapped_function: Callable) -> Callable:
            cls.registry[event] = [*cls.registry.get(event, []), wrapped_function]
            return wrapped_function

        return inner_wrapper

    @classmethod
    def merge(cls, other_registry: Dict[Type[Event], List[Callable]]) -> None:
        """Merge handlers from another registry, skipping duplicates by handler name."""
        for event, handlers in other_registry.items():
            existing_handlers = cls.registry.get(event, [])
            existing_names = {getattr(h, "__name__", id(h)) for h in existing_handlers}
            new_handlers = [
                h
                for h in handlers
                if getattr(h, "__name__", id(h)) not in existing_names
            ]
            cls.registry[event] = existing_handlers + new_handlers
