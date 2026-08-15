from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from zav.message_bus import HandlerException

from zav.agents_sdk.domain.chat_message import ChatMessage, ConversationContext

# The states of a conversation's latest turn. Defined once here so the domain
# model, the API DTOs, and ``append_messages`` agree on the same four values.
LastTurnStatus = Literal["running", "complete", "errored", "cancelled"]


class ConversationNotOwnedError(HandlerException):
    """A turn tried to write a conversation the requester does not own.

    Part of the ``append_messages`` contract: a store keys a conversation by
    something owner-agnostic (so it can be read when shared), so a requester who
    learned a shared conversation's ``session_id`` could otherwise append to —
    and thus "resume" — its owner's transcript. Sharing is read-only; only the
    owner writes. Stores raise this instead of writing; the app maps it to 403
    (see ``zav.agents_sdk.exception_handlers``). Mirrors how the stream buffer
    raises ``ChatStreamBusyError``/``ChatStreamGoneError`` for its own contract.
    """

    def __init__(self, session_id: str, owner: str, requester: str) -> None:
        self.session_id = session_id
        self.owner = owner
        self.requester = requester
        super().__init__(
            f"Conversation {session_id!r} is owned by {owner!r}; "
            f"{requester!r} cannot write to it (sharing is read-only)."
        )


class ChatConversationStore(ABC):
    """Durable transcript of a stateful chat — one row per conversation
    (``session_id``).

    The streaming recording/supervision rows are ephemeral per-turn control
    state; this store holds the conversation a client lists and reopens. The
    REST endpoints (list/get/delete) and the note mirroring read it directly;
    the chat handler only ever appends to it via ``append_messages``.
    """

    @abstractmethod
    async def append_messages(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        agent_identifier: str,
        messages: List[ChatMessage],
        user_roles: Optional[str] = None,
        user_tenants: Optional[str] = None,
        user_agent_id: Optional[str] = None,
        bot_params: Optional[Dict[str, Any]] = None,
        last_turn_status: Optional[LastTurnStatus] = None,
        create_if_absent: bool = True,
        conversation_context: Optional[ConversationContext] = None,
    ) -> None:
        """Open the conversation (create if absent, seeded with
        ``agent_identifier``) and append ``messages``, persisting.

        ``create_if_absent`` is ``True`` for the user turn (which opens the
        conversation) and ``False`` for the bot turn: a turn whose conversation
        was deleted mid-stream must not resurrect it as a ghost row.

        A conversation is keyed by (tenant, session_id) — ``session_id`` is a
        UUID, addressable on its own — with ``requester_uuid`` as the owner. A
        missing ``requester_uuid`` normalizes to "".

        ``user_roles`` and ``user_tenants`` are the raw service-deps header
        values from the originating request; they are forwarded to any
        downstream services (e.g. the notes API) that need them for auth.

        Raises ``ConversationNotOwnedError`` if ``requester_uuid`` is not the
        conversation's owner (writes are owner-only; sharing is read-only).
        """
        raise NotImplementedError

    @abstractmethod
    async def set_last_turn_status(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
        last_turn_status: LastTurnStatus,
    ) -> None:
        """Update only the conversation's ``last_turn_status`` (no message
        append) — e.g. to record a turn that ``errored`` instead of completing,
        so the list view does not show a permanently ``running`` turn.

        A no-op if the conversation does not exist. Raises
        ``ConversationNotOwnedError`` if ``requester_uuid`` is not the owner —
        the same owner-only write contract as ``append_messages``.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_transcript(
        self,
        tenant: str,
        requester_uuid: Optional[str],
        session_id: str,
    ) -> List[ChatMessage]:
        """Return the conversation's durable transcript, or ``[]`` if no row
        exists yet.

        The chat handler reads this for the stateful turn of an agent that
        restores no working state of its own: such an agent is sent only the new
        turn by the client, so the handler feeds it the stored transcript instead
        (see ``resolve_turn_conversation``). Keyed by (tenant, session_id) like
        ``append_messages``; reads are allowed for shared conversations, so this
        does not enforce the owner-only rule that writes do.
        """
        raise NotImplementedError
