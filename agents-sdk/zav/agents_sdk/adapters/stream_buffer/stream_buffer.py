import asyncio
from typing import AsyncGenerator, Callable, Dict, List, Optional, Tuple

from zav.logging import logger

from zav.agents_sdk.domain.chat_agent import ChatMessage

DEFAULT_INACTIVITY_SECONDS = 30
DEFAULT_CLEANUP_SECONDS = 120
DEFAULT_MAX_EVENTS_PER_BUFFER = 10_000
DEFAULT_MAX_BUFFERS_PER_TENANT = 50


class StreamBufferLimitError(Exception):
    pass


class StreamBuffer:
    def __init__(
        self,
        message_id: str,
        tenant: str,
        generator: AsyncGenerator[ChatMessage, None],
        inactivity_seconds: float = DEFAULT_INACTIVITY_SECONDS,
        cleanup_seconds: float = DEFAULT_CLEANUP_SECONDS,
        max_events: int = DEFAULT_MAX_EVENTS_PER_BUFFER,
        on_cleanup: Optional[Callable[[], None]] = None,
    ):
        self.__message_id = message_id
        self.__tenant = tenant
        self.__generator = generator
        self.__events: List[ChatMessage] = []
        self.__done = False
        self.__cancelled = False
        self.__error: Optional[Exception] = None
        self.__condition = asyncio.Condition()
        self.__inactivity_seconds = inactivity_seconds
        self.__cleanup_seconds = cleanup_seconds
        self.__max_events = max_events
        self.__on_cleanup = on_cleanup
        self.__cleanup_handle: Optional[asyncio.TimerHandle] = None
        self.__inactivity_handle: Optional[asyncio.TimerHandle] = None
        self.__consumer_count = 0
        self.__task = asyncio.create_task(self.__consume())
        self.__reset_inactivity_timer()

    @property
    def message_id(self) -> str:
        return self.__message_id

    @property
    def tenant(self) -> str:
        return self.__tenant

    @property
    def done(self) -> bool:
        return self.__done

    @property
    def cancelled(self) -> bool:
        return self.__cancelled

    @property
    def error(self) -> Optional[Exception]:
        return self.__error

    @property
    def events(self) -> List[ChatMessage]:
        return list(self.__events)

    async def cancel(self) -> None:
        if self.__done:
            return
        self.__cancel_inactivity_timer()
        self.__task.cancel()
        try:
            await self.__task
        except asyncio.CancelledError:
            pass
        await self.__generator.aclose()

    def __reset_inactivity_timer(self) -> None:
        self.__cancel_inactivity_timer()
        if self.__done:
            return
        loop = asyncio.get_running_loop()
        self.__inactivity_handle = loop.call_later(
            self.__inactivity_seconds,
            lambda: asyncio.ensure_future(self.__inactivity_timeout()),
        )

    def __cancel_inactivity_timer(self) -> None:
        if self.__inactivity_handle is not None:
            self.__inactivity_handle.cancel()
            self.__inactivity_handle = None

    async def __inactivity_timeout(self) -> None:
        if self.__done or self.__consumer_count > 0:
            return
        logger.info(
            f"Stream {self.__message_id} cancelled due to inactivity "
            f"(no consumers for {self.__inactivity_seconds}s)"
        )
        await self.cancel()

    async def __consume(self) -> None:
        try:
            async for message in self.__generator:
                async with self.__condition:
                    self.__events.append(message)
                    self.__condition.notify_all()
                if len(self.__events) >= self.__max_events:
                    logger.warning(
                        f"Stream {self.__message_id} reached max events "
                        f"({self.__max_events}), cancelling"
                    )
                    await self.__generator.aclose()
                    break
        except asyncio.CancelledError:
            async with self.__condition:
                self.__cancelled = True
                self.__condition.notify_all()
            raise
        except Exception as e:
            logger.error(
                f"Error consuming stream {self.__message_id}: {e}", exc_info=True
            )
            async with self.__condition:
                self.__error = e
                self.__condition.notify_all()
        finally:
            async with self.__condition:
                self.__done = True
                self.__condition.notify_all()
            self.__schedule_cleanup()

    def __schedule_cleanup(self) -> None:
        if self.__on_cleanup is None:
            return
        loop = asyncio.get_running_loop()
        self.__cleanup_handle = loop.call_later(
            self.__cleanup_seconds,
            self.__on_cleanup,
        )

    async def iter_events(
        self, start_index: int = 0
    ) -> AsyncGenerator[Tuple[int, ChatMessage], None]:
        self.__consumer_count += 1
        self.__cancel_inactivity_timer()
        try:
            index = start_index
            while True:
                async with self.__condition:
                    while index >= len(self.__events) and not self.__done:
                        await self.__condition.wait()

                while index < len(self.__events):
                    yield index, self.__events[index]
                    index += 1

                if self.__done and index >= len(self.__events):
                    if self.__error:
                        raise self.__error
                    return
        finally:
            self.__consumer_count -= 1
            if self.__consumer_count == 0 and not self.__done:
                self.__reset_inactivity_timer()


class StreamBufferRegistry:
    def __init__(
        self,
        max_buffers_per_tenant: int = DEFAULT_MAX_BUFFERS_PER_TENANT,
    ) -> None:
        self.__buffers: Dict[str, StreamBuffer] = {}
        self.__max_buffers_per_tenant = max_buffers_per_tenant

    @staticmethod
    def __key(tenant: str, requester_uuid: Optional[str], message_id: str) -> str:
        return f"{tenant}:{requester_uuid or ''}:{message_id}"

    def create(
        self,
        message_id: str,
        tenant: str,
        generator: AsyncGenerator[ChatMessage, None],
        requester_uuid: Optional[str] = None,
    ) -> StreamBuffer:
        tenant_count = sum(1 for k in self.__buffers if k.startswith(f"{tenant}:"))
        if tenant_count >= self.__max_buffers_per_tenant:
            raise StreamBufferLimitError(
                f"Tenant {tenant} has reached the maximum number of "
                f"concurrent streams ({self.__max_buffers_per_tenant})"
            )
        buffer = StreamBuffer(
            message_id=message_id,
            tenant=tenant,
            generator=generator,
            on_cleanup=lambda: self.remove(
                message_id=message_id,
                tenant=tenant,
                requester_uuid=requester_uuid,
            ),
        )
        key = self.__key(tenant, requester_uuid, message_id)
        logger.info(
            f"StreamBuffer created: key={key} "
            f"(tenant={tenant}, requester_uuid={requester_uuid}, "
            f"message_id={message_id})"
        )
        self.__buffers[key] = buffer
        return buffer

    def get(
        self,
        message_id: str,
        tenant: str,
        requester_uuid: Optional[str] = None,
    ) -> Optional[StreamBuffer]:
        key = self.__key(tenant, requester_uuid, message_id)
        buffer = self.__buffers.get(key)
        if buffer is None:
            logger.warning(
                f"StreamBuffer get: not found for key={key}. "
                f"Active keys: {list(self.__buffers.keys())}"
            )
        else:
            logger.info(f"StreamBuffer get: found key={key}")
        return buffer

    def remove(
        self,
        message_id: str,
        tenant: str,
        requester_uuid: Optional[str] = None,
    ) -> None:
        key = self.__key(tenant, requester_uuid, message_id)
        removed = self.__buffers.pop(key, None)
        if removed is not None:
            logger.info(f"StreamBuffer removed: key={key}")
        else:
            logger.warning(f"StreamBuffer remove: not found for key={key}")

    async def cancel_stream(
        self,
        message_id: str,
        tenant: str,
        requester_uuid: Optional[str] = None,
    ) -> bool:
        key = self.__key(tenant, requester_uuid, message_id)
        buffer = self.__buffers.get(key)
        if buffer is None:
            logger.warning(
                f"cancel_stream: buffer not found for key={key} "
                f"(tenant={tenant}, requester_uuid={requester_uuid}, "
                f"message_id={message_id}). "
                f"Active keys: {list(self.__buffers.keys())}"
            )
            return False
        await buffer.cancel()
        self.__buffers.pop(key, None)
        logger.info(f"StreamBuffer cancelled: key={key}")
        return True

    def clear(self) -> None:
        self.__buffers.clear()
