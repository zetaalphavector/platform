import asyncio
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, List, Optional

from zav.agents_sdk.domain.chat_message import ChatMessage


class BackgroundTurn:
    """A streamed turn run off the Streamlit script thread. Streamlit reruns the
    script on every click and cancels what the previous run was doing, so a
    turn that waits for an approval or reads messages sent while it runs must
    outlive the run that started it."""

    def __init__(self, stream: Callable[[], AsyncIterator[ChatMessage]]):
        self.__stream = stream
        self.last_message: Optional[ChatMessage] = None
        self.error: Optional[Exception] = None
        self.__done = threading.Event()
        threading.Thread(
            target=asyncio.run, args=(self.__drain(),), daemon=True
        ).start()

    @property
    def done(self) -> bool:
        return self.__done.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self.__done.wait(timeout)

    async def __drain(self) -> None:
        try:
            async for message in self.__stream():
                self.last_message = message
        except Exception as e:
            self.error = e
        finally:
            self.__done.set()


@dataclass
class RunningTurn:
    """The dev UI's turn in flight, and where its conversation's inbox is."""

    turn: BackgroundTurn
    tenant: str
    requester_uuid: Optional[str]
    session_id: str
    debug_storage: List[Any]
