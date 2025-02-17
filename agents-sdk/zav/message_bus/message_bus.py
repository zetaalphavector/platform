import random
from asyncio import sleep
from typing import Callable, Dict, List, Type

from zav.logging import logger

from zav.message_bus.common import Command, Event, Message
from zav.message_bus.errors import NonRetryableHandlerError, RetryableHandlerError


class MessageBus:
    def __init__(
        self,
        command_handlers: Dict[Type[Command], Callable],
        event_handlers: Dict[Type[Event], List[Callable]],
        exception_handlers: Dict[Type[Exception], Callable],
    ):

        self.__command_handlers = command_handlers
        self.__event_handlers = event_handlers
        self.__exception_handlers = exception_handlers

    async def handle(self, message: Message, retry_attempts=0):

        try:
            result = await self.__handle_message(message)
            return result
        except Exception as e:
            exc_type = type(e)
            if exc_type in self.__exception_handlers:
                handled_exception = self.__exception_handlers[exc_type](e)
                try:
                    raise handled_exception
                except RetryableHandlerError as retryable_error:
                    if retry_attempts >= retryable_error.max_retries:
                        logger.exception(
                            f"Message {message.__class__.__name__} failed after "
                            f"{retry_attempts} retries. "
                            f"Error: {e}"
                        )
                        raise NonRetryableHandlerError(e)
                    attempt = retry_attempts + 1
                    delay = attempt * retryable_error.base_delay
                    # Add random jitter to delay based on its value
                    jitter = random.uniform(-0.1 * delay, 0.1 * delay)
                    delay += jitter
                    logger.info(
                        (
                            f"Retry attempt #{attempt} for message "
                            f"{message.__class__.__name__}. "
                            "Will retry in {delay} seconds."
                            f"Error: {e}"
                        )
                    )
                    await sleep(delay)
                    return await self.handle(message, attempt)
                except Exception as non_retryable_exception:
                    logger.exception(f"Non retryable error: {non_retryable_exception}")
                    raise e
            else:
                logger.exception(f"No exception handler for {exc_type}")
                raise e

    async def __handle_message(self, message: Message):
        results = []
        queue = [message]
        while queue:
            message = queue.pop(0)
            if isinstance(message, Event):
                await self.__handle_event(message, queue)
            elif isinstance(message, Command):
                cmd_result = await self.__handle_command(message, queue)
                results.append(cmd_result)
            else:
                raise Exception(f"{message} was not an Event or Command")
        return results

    async def __handle_event(self, event: Event, queue: List[Message]):
        for handler in self.__event_handlers[type(event)]:
            logger.debug(f"handling event {event} with handler {handler}")
            await handler(event, queue)

    async def __handle_command(self, command: Command, queue: List[Message]):
        logger.debug(f"handling command {command}")
        handler = self.__command_handlers[type(command)]
        result = await handler(command, queue)
        return result
