from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from zav.agents_sdk.domain.chat_agent import ChatMessage


@dataclass
class RecordedStreamEvent:
    index: int
    message: ChatMessage
    # Which turn this snapshot belongs to. The record is keyed by the
    # conversation, so a follower must check the occupant: a different
    # ``message_id`` means the conversation moved on to a newer turn and the
    # followed one is gone.
    message_id: str
    # The producer's supervision-lease epoch. ``put`` refuses a write whose
    # epoch is below the record's current one, so a producer fenced by a newer
    # owner cannot overwrite the new owner's snapshot.
    epoch: int


class ChatStreamRecordingStore(ABC):
    """Durable snapshot of a conversation's in-progress assistant turn.

    One overwritten record per conversation (``session_id``), holding the
    latest cumulative ``ChatMessage`` of whichever turn currently occupies the
    slot, that turn's ``message_id``, and the producer's yield ``index`` at
    that point. Each streamed message already contains the whole answer so
    far, so keeping only the newest is loss-free — a reconnecting pod reads
    this single snapshot and paces out whatever content the consumer has not
    seen yet, diffing by ``index``.

    Keying by conversation makes the next turn's first write the garbage
    collection of the previous turn's snapshot: a conversation occupies one
    slot forever instead of leaking a record per turn. ``delete`` exists only
    for the no-next-turn fallback, guarded by ``message_id`` so a reaper for a
    finished turn can never remove a successor's snapshot.

    The production backend is accumulated state in object storage — one blob
    overwritten per conversation, not an append log — so the contract is
    put-latest / read-latest, fenced by ``epoch``: ``put`` rejects a write whose
    epoch is below the stored record's, so a producer that lost the supervision
    lease to a newer owner cannot overwrite it. This is the same compare-and-swap
    the supervision lease uses, applied to the record's own row (a cross-row
    check would not be protected by the database's snapshot isolation).
    """

    @abstractmethod
    async def put(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        event: RecordedStreamEvent,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def read(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> Optional[RecordedStreamEvent]:
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        message_id: str,
    ) -> None:
        """Remove the conversation's snapshot iff it still belongs to
        ``message_id``. A snapshot already overwritten by a newer turn is left
        untouched — overwriting was the cleanup."""
        raise NotImplementedError
