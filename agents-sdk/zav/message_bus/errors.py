from typing import Any, Callable, Dict, Type, Union


class HandlerException(Exception):
    """Base exception for handler errors."""


class NonRetryableHandlerError(Exception):
    pass


class RetryableHandlerError(Exception):
    def __init__(
        self,
        original_exception: Exception,
        max_retries: int,
        base_delay: float = 0.1,
    ):
        self.original_exception = original_exception
        self.max_retries = max_retries
        self.base_delay = base_delay


HandlerExceptionHandler = Callable[
    [Any], Union[NonRetryableHandlerError, RetryableHandlerError]
]


class ExceptionHandlerRegistry:
    registry: Dict[Type[Exception], HandlerExceptionHandler] = {}

    @classmethod
    def register(
        cls, exception: Type[Exception]
    ) -> Callable[[HandlerExceptionHandler], HandlerExceptionHandler]:
        def inner_wrapper(
            wrapped_function: HandlerExceptionHandler,
        ) -> HandlerExceptionHandler:
            cls.registry[exception] = wrapped_function
            return wrapped_function

        return inner_wrapper
