import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from zav.agents_sdk.domain.chat_message import ChatMessage, ContentPartTool
from zav.agents_sdk.domain.message_inbox import (
    InboxItem,
    MessageInboxStore,
    ToolDecision,
)

# How often a waiting turn looks at the inbox, as a follower polls the stream
# recording.
DEFAULT_POLL_INTERVAL_SECONDS = 0.3


DEFAULT_APPROVAL_TIMEOUT_SECONDS = 600.0
# Several sub-agents can wait at once; each re-checks what the others took at
# least this often.
_DECISION_WAIT_SLICE_SECONDS = 1.0


@dataclass(frozen=True)
class ApprovalOutcome:
    decisions: Dict[str, ToolDecision]
    # The user sent a message instead of deciding.
    superseded: bool


class TurnInbox:
    """The interactive turn's reader of the conversation's inbox, as the turn
    that holds the conversation's lease. Every message reaches a turn through
    here, the first one included. When the agent reads queued messages, those
    sent mid-turn go to the model before its next call; decisions go to the
    calls waiting for them, the sub-agents' included.
    A sub-agent's approval cards are published here and shown in the turn's
    output by the root agent's tool runner."""

    def __init__(
        self,
        store: MessageInboxStore,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        approval_timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self.__store = store
        self.__tenant = tenant
        self.__requester_uuid = requester_uuid
        self.__session_id = session_id
        self.__approval_timeout_seconds = approval_timeout_seconds
        self.__poll_interval_seconds = poll_interval_seconds
        self.__messages: List[ChatMessage] = []
        self.__messages_received = 0
        self.__decisions: Dict[str, ToolDecision] = {}
        self.__take_lock = asyncio.Lock()
        self.__card_sink: Optional["asyncio.Queue[ContentPartTool]"] = None

    async def messages(self) -> List[ChatMessage]:
        async with self.__take_lock:
            await self.__take(0)
        taken, self.__messages = self.__messages, []
        return taken

    async def has_messages(self) -> bool:
        """Whether a message waits for the model; it stays for ``messages``."""
        async with self.__take_lock:
            await self.__take(0)
        return bool(self.__messages)

    async def await_decisions(self, tool_call_ids: List[str]) -> ApprovalOutcome:
        deadline = time.monotonic() + self.__approval_timeout_seconds
        messages_before = self.__messages_received
        waiting = set(tool_call_ids)
        decisions: Dict[str, ToolDecision] = {}
        while True:
            for tool_call_id in list(waiting):
                if tool_call_id in self.__decisions:
                    decisions[tool_call_id] = self.__decisions.pop(tool_call_id)
                    waiting.discard(tool_call_id)
            superseded = self.__messages_received > messages_before
            remaining = deadline - time.monotonic()
            if not waiting or superseded or remaining <= 0:
                return ApprovalOutcome(decisions=decisions, superseded=superseded)
            async with self.__take_lock:
                if not waiting & self.__decisions.keys():
                    await self.__take(min(remaining, _DECISION_WAIT_SLICE_SECONDS))

    async def return_unread(self) -> None:
        """Puts the messages this turn took but no model read back in the
        inbox, ahead of those sent since, for the next turn. An agent that
        does not read queued messages still takes them while it waits for an
        approval, since a message declines the calls waiting."""
        async with self.__take_lock:
            await self.__take(0)
            unread, self.__messages = self.__messages, []
            for message in unread:
                await self.__store.append(
                    tenant=self.__tenant,
                    requester_uuid=self.__requester_uuid,
                    session_id=self.__session_id,
                    item=InboxItem(message=message),
                )

    def publish(self, cards: List[ContentPartTool]) -> bool:
        """False when no root agent is there to show them."""
        if self.__card_sink is None:
            return False
        for card in cards:
            self.__card_sink.put_nowait(card)
        return True

    @contextmanager
    def showing_cards_in(
        self, sink: "asyncio.Queue[ContentPartTool]"
    ) -> Iterator[None]:
        self.__card_sink = sink
        try:
            yield
        finally:
            self.__card_sink = None

    async def __take(self, wait_seconds: float) -> None:
        for item in await self.__poll(wait_seconds):
            if item.message is not None:
                self.__messages.append(item.message)
                self.__messages_received += 1
            if item.tool_decision is not None:
                self.__decisions[item.tool_decision.tool_call_id] = item.tool_decision

    async def __poll(self, wait_seconds: float) -> List[InboxItem]:
        """Items added since the last take, oldest first. With nothing new,
        waits up to ``wait_seconds`` for one."""
        deadline = time.monotonic() + wait_seconds
        while True:
            items = await self.__store.take(
                tenant=self.__tenant,
                requester_uuid=self.__requester_uuid,
                session_id=self.__session_id,
            )
            remaining = deadline - time.monotonic()
            if items or remaining <= 0:
                return items
            await asyncio.sleep(min(self.__poll_interval_seconds, remaining))
