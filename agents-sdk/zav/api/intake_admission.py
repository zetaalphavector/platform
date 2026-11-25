import asyncio
from collections import deque
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import AsyncIterator, Callable, Deque, Optional, Tuple

from fastapi import HTTPException

from zav.api.app_resources import AppResource

GIB = 1024**3

# Receiving a JSON body through JsonBodyRoute takes 2 bytes of memory for every
# byte of the body: the body and the objects parsed from it are alive together.
# Measured as the peak RSS of pipeline-service's API process over the request's
# Content-Length, while it reads, validates, decodes and stores base64 documents:
# 1.99-2.02x for one document of 32 or 64 MiB, for batches of 4 to 100 documents
# and for a body the model rejects, 2026-11-15. The part over 2x is at most
# 1.6 MiB per request, whatever the size of the body.
INTAKE_COST_PER_BODY_BYTE = 2

INTAKE_WAIT_SECONDS = 60.0


def intake_cost(content_length: int) -> int:
    """The memory counted for a request whose body is `content_length` long."""
    return content_length * INTAKE_COST_PER_BODY_BYTE


class ByteBudget:
    """Bytes a process lends to requests in flight. Freed bytes go to the oldest
    waiter first, and a new request waits behind every older waiter even when it
    would fit, so a large request is never starved by small ones."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.__held = 0
        self.__waiters: Deque[Tuple[int, asyncio.Future[None]]] = deque()

    @property
    def held(self) -> int:
        return self.__held

    async def acquire(self, size: int) -> None:
        if size > self.capacity:
            raise ValueError(f"{size} bytes exceed the budget of {self.capacity}")
        if not self.__waiters and self.__held + size <= self.capacity:
            self.__held += size
            return
        waiter = asyncio.get_running_loop().create_future()
        entry = (size, waiter)
        self.__waiters.append(entry)
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.cancelled():
                if entry in self.__waiters:
                    self.__waiters.remove(entry)
                self.__wake()
            else:
                # Granted, then cancelled before it resumed.
                self.release(size)
            raise

    def release(self, size: int) -> None:
        self.__held -= size
        self.__wake()

    def __wake(self) -> None:
        while self.__waiters:
            size, waiter = self.__waiters[0]
            if waiter.done():
                self.__waiters.popleft()
                continue
            if self.__held + size > self.capacity:
                return
            self.__waiters.popleft()
            self.__held += size
            waiter.set_result(None)


class IntakeAdmission:
    """Admits a request body into the process only when its cost fits the intake
    budget, before a byte of it is read: uvicorn stops reading a socket whose body
    the app is not consuming, so a waiting request holds no memory. `setting` is
    the name the service's operator sets the budget under; a refusal names it."""

    def __init__(
        self,
        budget: ByteBudget,
        setting: str,
        wait_seconds: float = INTAKE_WAIT_SECONDS,
        timeout: Callable[
            [float], AbstractAsyncContextManager[object]
        ] = asyncio.timeout,
    ) -> None:
        self.__budget = budget
        self.__setting = setting
        self.__wait_seconds = wait_seconds
        self.__timeout = timeout

    @classmethod
    def from_memory_gb(cls, memory_gb: float, setting: str) -> "IntakeAdmission":
        return cls(ByteBudget(int(memory_gb * GIB)), setting)

    @asynccontextmanager
    async def admitted(self, content_length: Optional[int]) -> AsyncIterator[None]:
        if content_length is None:
            raise HTTPException(
                status_code=411, detail="A request body needs a Content-Length."
            )
        cost = intake_cost(content_length)
        if cost > self.__budget.capacity:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"The request needs {cost / GIB:.2f} GB of intake memory and "
                    f"the service has {self.__setting}="
                    f"{self.__budget.capacity / GIB:g}. Send less in one request, "
                    "or raise the API container's memory limit and "
                    f"{self.__setting} together."
                ),
            )
        try:
            async with self.__timeout(self.__wait_seconds):
                await self.__budget.acquire(cost)
        except TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="The service is busy receiving other requests.",
                headers={"Retry-After": "1"},
            ) from None
        try:
            yield
        finally:
            self.__budget.release(cost)


get_intake_admission = AppResource[IntakeAdmission]("intake_admission")
