import asyncio
from typing import Awaitable, Callable

from zav.logging import logger


class AgentAsyncWorker:
    """Periodically calls the provided async worker function at the specified interval,
    until stopped. The worker function is also called one final time when finish
    is called, to ensure any final work is done before stopping."""

    def __init__(
        self,
        worker_func: Callable[[], Awaitable[None]],
        interval_seconds: float = 5.0,
    ):
        self.__worker_func = worker_func
        self.__interval_seconds = interval_seconds
        self.__stop_event = asyncio.Event()
        self.__is_done = False
        self.__task = asyncio.create_task(self.__run())

    async def finish(self) -> None:
        """Signals the worker to stop and waits for it to finish."""
        if self.__is_done:
            return None
        self.__is_done = True
        self.__stop_event.set()
        await self.__task
        await self.__call_worker_func()

    async def stop(self) -> None:
        """Signals the worker to stop without the final call ``finish`` makes.

        Used by the supervision heartbeat, where one extra tick after the turn
        has been sealed would overwrite the terminal status with a fresh
        ``running`` and make a finished turn look alive again.
        """
        if self.__is_done:
            return None
        self.__is_done = True
        self.__stop_event.set()
        await self.__task

    async def __run(self) -> None:
        while not self.__stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self.__stop_event.wait(), timeout=self.__interval_seconds
                )
            except asyncio.TimeoutError:
                await self.__call_worker_func()

    async def __call_worker_func(self) -> None:
        try:
            await self.__worker_func()
        except Exception as e:
            # Log the exception, but don't let it crash the worker
            logger.exception(f"Exception in AgentAsyncWorker: {e}")
