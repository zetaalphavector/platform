from typing import Callable, Dict, List, Type

from zav.message_bus.common import Command, Event


class CommandHandlerRegistry:

    registry: Dict[Type[Command], Callable] = {}

    @classmethod
    def register(cls, command: Type[Command]) -> Callable:
        def inner_wrapper(wrapped_function: Callable) -> Callable:
            cls.registry[command] = wrapped_function
            return wrapped_function

        return inner_wrapper


class EventHandlerRegistry:

    registry: Dict[Type[Event], List[Callable]] = {}

    @classmethod
    def register(cls, event: Type[Event]) -> Callable:
        def inner_wrapper(wrapped_function: Callable) -> Callable:
            cls.registry[event] = [*cls.registry.get(event, []), wrapped_function]
            return wrapped_function

        return inner_wrapper
