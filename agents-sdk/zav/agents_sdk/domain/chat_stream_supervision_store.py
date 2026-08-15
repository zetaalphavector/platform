from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ChatStreamStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    ERRORED = "errored"


@dataclass
class ChatStreamSupervision:
    status: ChatStreamStatus
    heartbeat_at: float
    # Which turn currently occupies the conversation's slot. The row is keyed
    # by the conversation, so readers and claimants must check the occupant:
    # a claimant refusing a fresh RUNNING row with a *different* message_id is
    # what makes the slot a conversation mutex, and a follower seeing the
    # message_id change mid-follow knows its turn is gone.
    message_id: str
    # Monotonic lease token. Whoever produces the conversation's current turn
    # owns the slot at this epoch; a pod takes the slot over (after the prior
    # producer's pod died, or to resume the same turn) by swapping in a higher
    # epoch, which fences the old producer out.
    epoch: int = 0


class ChatStreamSupervisionStore(ABC):
    """Liveness record and ownership lease for a conversation's in-flight
    assistant turn.

    The supervision twin of the stream recording: a single mutable row per
    conversation (``session_id``) that answers "which turn is being produced
    now, is the producing pod still alive, who owns the slot, and how did the
    turn end?". The owner overwrites it on a timer (``heartbeat_at``) and on
    terminal transitions (``status``); a reconnecting pod reads it to tell a
    turn that is still being produced apart from one whose pod died.

    The two are deliberately separate stores. Both are single overwritten
    blobs, but they carry different state: recording holds the answer itself —
    the latest cumulative snapshot, kept as a conversation message — while
    supervision holds a tiny control row. Keeping them apart keeps heartbeats
    cheap (they never rewrite the transcript) and lets the answer outlive the
    lease.

    Keying by conversation makes the row double as the conversation's send
    mutex: a new turn claims the slot by overwriting it (the overwrite IS the
    cleanup of the previous turn's row), and a claim against a fresh RUNNING
    row holding a different turn must be refused — one in-flight turn per
    conversation. ``delete`` exists only for the no-next-turn fallback,
    guarded so a reaper can only remove the row while its own turn still
    occupies it.

    Liveness is decided by the *reader*, not stored: ``dead`` means a
    ``RUNNING`` row whose ``heartbeat_at`` has gone stale, never a flag the
    producer writes (a killed pod gets no chance to write anything).

    Ownership is the lease: every write goes through ``compare_and_swap`` on
    ``epoch``, so a pod that takes a presumed-dead turn over claims a higher
    epoch and the old producer — if it was only slow, not dead — self-fences
    the moment its own swaps start failing. Two pods never own one slot.
    """

    @abstractmethod
    async def compare_and_swap(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        expected_epoch: Optional[int],
        supervision: ChatStreamSupervision,
    ) -> bool:
        """Atomically store ``supervision`` iff the current row's ``epoch``
        equals ``expected_epoch`` — or there is no row yet and
        ``expected_epoch`` is None. Returns True on write, False when the
        precondition fails (the caller lost the create race or has been fenced
        by a newer owner and must stop producing)."""
        raise NotImplementedError

    @abstractmethod
    async def read(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> Optional[ChatStreamSupervision]:
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        message_id: str,
        epoch: int,
    ) -> None:
        """Remove the conversation's row iff it still holds ``message_id`` at
        ``epoch``. A row already claimed by a newer turn (or a newer epoch of
        the same turn) is left untouched — the claim was the cleanup."""
        raise NotImplementedError
