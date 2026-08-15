import asyncio
from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

from zav.logging import logger

from zav.agents_sdk.domain.chat_agent import ChatMessage

DEFAULT_INACTIVITY_SECONDS = 30
DEFAULT_CLEANUP_SECONDS = 120
DEFAULT_MAX_EVENTS_PER_BUFFER = 10_000
DEFAULT_MAX_BUFFERS_PER_TENANT = 50


class StreamBufferLimitError(Exception):
    pass


class StreamSource(Protocol):
    """What a chat-stream consumer needs: an id, indexed events, and a
    terminal verdict.

    Both the live ``LiveStreamSource`` and the durable
    ``RecordingReplaySource`` satisfy this, so the controller streams either
    without caring which pod is producing. ``terminal_reason`` is set once
    iteration ends: ``None`` for a live source (a clean close already means
    "done"), or one of ``done`` / ``errored`` / ``dead`` for a followed
    recording.
    """

    @property
    def message_id(self) -> str: ...

    @property
    def terminal_reason(self) -> Optional[str]: ...

    def iter_events(
        self, start_index: int = 0
    ) -> AsyncGenerator[Tuple[int, ChatMessage], None]: ...


class RegisteredStream(StreamSource, Protocol):
    """A live, cancellable source the registry owns for the duration of an
    in-flight turn.

    Adds ``aclose`` on top of the read surface: stopping production, used both
    by an explicit cancel and by the buffer's inactivity timeout.
    """

    async def aclose(self) -> None: ...


class StreamBuffer:
    """Replay log for one in-flight assistant turn.

    Pure data: holds the ordered event list, terminal flags, and a
    condition variable for live consumers. Does NOT own the producer
    task — the owning ``LiveStreamSource`` runs the producer and pushes
    events in via ``append`` / terminal flags via ``mark_*``.

    Lifetime helpers (inactivity, cleanup) are scheduled here because
    they are driven by consumer activity on ``iter_events``, but the
    actual effects — "stop producing" and "forget this turn" — are
    delegated through the ``on_inactive`` / ``on_cleanup`` callbacks
    supplied by the owner.
    """

    def __init__(
        self,
        message_id: str,
        tenant: str,
        on_inactive: Callable[[], Awaitable[None]],
        on_cleanup: Optional[Callable[[], None]] = None,
        inactivity_seconds: float = DEFAULT_INACTIVITY_SECONDS,
        cleanup_seconds: float = DEFAULT_CLEANUP_SECONDS,
        max_events: int = DEFAULT_MAX_EVENTS_PER_BUFFER,
    ):
        self.__message_id = message_id
        self.__tenant = tenant
        self.__events: List[ChatMessage] = []
        self.__done = False
        self.__cancelled = False
        self.__error: Optional[Exception] = None
        self.__condition = asyncio.Condition()
        self.__on_inactive = on_inactive
        self.__on_cleanup = on_cleanup
        self.__inactivity_seconds = inactivity_seconds
        self.__cleanup_seconds = cleanup_seconds
        self.__max_events = max_events
        self.__consumer_count = 0
        self.__inactivity_handle: Optional[asyncio.TimerHandle] = None
        self.__cleanup_handle: Optional[asyncio.TimerHandle] = None
        self.__reset_inactivity_timer()

    @property
    def message_id(self) -> str:
        return self.__message_id

    @property
    def terminal_reason(self) -> Optional[str]:
        # A live buffer carries no out-of-band verdict: a clean stream close
        # already means the turn finished. Only the followed recording needs
        # to distinguish "done" from "producer died".
        return None

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

    @property
    def max_events(self) -> int:
        return self.__max_events

    @property
    def event_count(self) -> int:
        return len(self.__events)

    @property
    def __terminated(self) -> bool:
        # Any terminal verdict releases a waiting consumer: a clean ``done``, a
        # producer error, or a cancellation. ``mark_error`` / ``mark_cancelled``
        # can land WITHOUT a following ``mark_done`` (e.g. a producer that fails
        # to seal), so the wait loop must treat them as terminal too — otherwise
        # a consumer parked in ``iter_events`` would wait forever.
        return self.__done or self.__error is not None or self.__cancelled

    async def append(self, message: ChatMessage) -> None:
        async with self.__condition:
            self.__events.append(message)
            self.__condition.notify_all()

    async def mark_done(self) -> None:
        if self.__done:
            return
        async with self.__condition:
            self.__done = True
            self.__condition.notify_all()
        self.__cancel_inactivity_timer()
        self.__schedule_cleanup()

    async def mark_error(self, error: Exception) -> None:
        async with self.__condition:
            self.__error = error
            self.__condition.notify_all()

    async def mark_cancelled(self) -> None:
        async with self.__condition:
            self.__cancelled = True
            self.__condition.notify_all()

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
            f"Stream {self.__message_id} inactive for "
            f"{self.__inactivity_seconds}s with no readers; invoking inactivity handler"
        )
        try:
            await self.__on_inactive()
        except Exception:
            logger.exception(f"Error in inactivity callback for {self.__message_id}")

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
                    while index >= len(self.__events) and not self.__terminated:
                        await self.__condition.wait()

                while index < len(self.__events):
                    yield index, self.__events[index]
                    index += 1

                if self.__terminated and index >= len(self.__events):
                    if self.__error:
                        raise self.__error
                    return
        finally:
            self.__consumer_count -= 1
            if self.__consumer_count == 0 and not self.__done:
                self.__reset_inactivity_timer()


class StreamBufferRegistry:
    """Pure registry of in-flight live streams.

    Holds one ``RegisteredStream`` per active turn, keyed by
    ``(tenant, requester_uuid, message_id)``, and enforces the per-tenant
    concurrency cap. It does NOT run the producer or own any task — each
    ``LiveStreamSource`` owns its own producer and timer tasks. The registry
    only records, hands back, and tears down entries, so "who is producing
    this turn" lives in one self-contained source object instead of being
    smeared across an entry, a task, and a producer factory.
    """

    def __init__(
        self,
        max_buffers_per_tenant: int = DEFAULT_MAX_BUFFERS_PER_TENANT,
    ) -> None:
        self.__entries: Dict[str, RegisteredStream] = {}
        self.__max_buffers_per_tenant = max_buffers_per_tenant

    @staticmethod
    def __key(tenant: str, requester_uuid: Optional[str], message_id: str) -> str:
        return f"{tenant}:{requester_uuid or ''}:{message_id}"

    def ensure_capacity(self, tenant: str) -> None:
        tenant_count = sum(1 for k in self.__entries if k.startswith(f"{tenant}:"))
        if tenant_count >= self.__max_buffers_per_tenant:
            raise StreamBufferLimitError(
                f"Tenant {tenant} has reached the maximum number of "
                f"concurrent streams ({self.__max_buffers_per_tenant})"
            )

    def register(
        self,
        message_id: str,
        tenant: str,
        source: RegisteredStream,
        requester_uuid: Optional[str] = None,
    ) -> None:
        key = self.__key(tenant, requester_uuid, message_id)
        if key in self.__entries:
            # message_id is a fresh per-turn UUID, so the full key is unique in
            # practice and this should not happen. Log loudly if it ever does:
            # the previous source's producer/timer tasks are not torn down here,
            # and its later on_cleanup is identity-guarded (see ``remove``) so it
            # cannot evict this newer occupant.
            logger.warning(
                f"StreamBuffer register: overwriting existing entry for key={key}"
            )
        self.__entries[key] = source
        logger.info(
            f"StreamBuffer created: key={key} "
            f"(tenant={tenant}, requester_uuid={requester_uuid}, "
            f"message_id={message_id})"
        )

    def get(
        self,
        message_id: str,
        tenant: str,
        requester_uuid: Optional[str] = None,
    ) -> Optional[RegisteredStream]:
        key = self.__key(tenant, requester_uuid, message_id)
        source = self.__entries.get(key)
        if source is None:
            logger.warning(
                f"StreamBuffer get: not found for key={key}. "
                f"Active keys: {list(self.__entries.keys())}"
            )
            return None
        logger.info(f"StreamBuffer get: found key={key}")
        return source

    def remove(
        self,
        message_id: str,
        tenant: str,
        requester_uuid: Optional[str] = None,
        expected_source: Optional[RegisteredStream] = None,
    ) -> None:
        key = self.__key(tenant, requester_uuid, message_id)
        current = self.__entries.get(key)
        if current is None:
            logger.warning(f"StreamBuffer remove: not found for key={key}")
            return
        if expected_source is not None and current is not expected_source:
            # A newer source has taken this key; don't let a stale source's
            # cleanup evict the live occupant.
            logger.info(
                f"StreamBuffer remove: skipped for key={key}; "
                f"entry already replaced by a newer source"
            )
            return
        self.__entries.pop(key, None)
        logger.info(f"StreamBuffer removed: key={key}")

    async def cancel_stream(
        self,
        message_id: str,
        tenant: str,
        requester_uuid: Optional[str] = None,
    ) -> bool:
        key = self.__key(tenant, requester_uuid, message_id)
        source = self.__entries.get(key)
        if source is None:
            logger.warning(
                f"cancel_stream: stream not found for key={key} "
                f"(tenant={tenant}, requester_uuid={requester_uuid}, "
                f"message_id={message_id}). "
                f"Active keys: {list(self.__entries.keys())}"
            )
            return False
        await source.aclose()
        # aclose() awaited above, so a newer source could have re-registered
        # this key meanwhile; only evict if it is still the one we cancelled.
        if self.__entries.get(key) is source:
            self.__entries.pop(key, None)
        logger.info(f"StreamBuffer cancelled: key={key}")
        return True

    def clear(self) -> None:
        self.__entries.clear()
