from typing import Any, Dict, List, Optional

from zav.pydantic_compat import PYDANTIC_V2, BaseModel, ConfigDict

from zav.agents_sdk.adapters.stream_buffer.resumable_chat_streams import (
    ChatStreamLiveness,
)
from zav.agents_sdk.domain import ChatMessage, ConversationContext, FunctionSpec


# NOTE: model docstrings become the schema ``description`` in the generated
# OpenAPI spec and leak into the public client SDKs, so keep this kind of
# implementation/usage detail as comments, not docstrings.
#
# Chat request body.
#
# ``session_id`` names a conversation (trace grouping, and in stateful mode the
# persisted-state key); ``message_id`` names one turn (stream key, per-message
# trace, feedback target). The body never carries ``message_id`` — every request
# is a new user turn whose id the service assigns and returns on the bot message.
# Every turn is assigned a ``session_id`` too — the one sent in the body, or a
# freshly minted one — and the bot message always carries both ids back. Echo the
# ``session_id`` on later requests to keep a conversation's traces grouped, in any
# mode; ``stateful`` only controls whether working state is persisted under it.
#
# Modes:
# - Stateless (default): omit ``stateful``. The service keeps nothing between
#   turns, so ``conversation`` must carry the full history on every request. Send
#   the prior bot message's ``session_id`` to keep the conversation's traces
#   grouped; it is used for tracing only, never to load or save state.
# - Stateful, first turn: send ``stateful=true`` and omit ``session_id``;
#   ``conversation`` carries the opening user message(s). The service mints a
#   fresh ``session_id`` (distinct from the turn's ``message_id``) and persists
#   the agent's working state under it.
# - Stateful, follow-up: send ``stateful=true``, the ``session_id`` from the
#   prior bot message, and only the new user message(s) in ``conversation``. The
#   session transcript is restored server-side and the incoming messages are
#   appended to it; re-sending the full history would duplicate it in the model
#   context.
# - Resume after a crash: POST with query ``resume=true`` and
#   ``message_id=<id of the interrupted turn>`` (it addresses an existing turn's
#   stream, like ``GET /chat/stream/{message_id}``), plus ``stateful=true`` and
#   the ``session_id`` in the body. The service takes that turn's stream over
#   (fencing the previous producer via the supervision lease) and re-drives it
#   from the session's last checkpoint, ignoring the body ``conversation``; with
#   no checkpoint yet it falls back to a fresh turn with the incoming message.
class ChatResponseForm(BaseModel):
    agent_identifier: str
    conversation: List[ChatMessage]
    conversation_context: Optional[ConversationContext] = None
    bot_params: Optional[Dict[str, Any]] = None
    user_agent_id: Optional[str] = None
    stateful: Optional[bool] = None
    session_id: Optional[str] = None


# Chat response body: the produced conversation plus optional function specs.
class ChatResponseItem(ChatResponseForm):
    function_specs: Optional[List[FunctionSpec]] = None

    if PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True)
    else:

        class Config:
            orm_mode = True


class ChatStreamItem(BaseModel):
    event: str
    id: str
    data: ChatMessage
    retry: int


# Which turn occupies a conversation's stream slot, and its state:
# ``running`` / ``done`` / ``errored`` / ``dead``.
class ChatStreamStatusItem(BaseModel):
    message_id: str
    status: ChatStreamLiveness


# Body of the 409 returned when a conversation already has a turn in flight:
# the occupant turn the client should attach to instead of starting another.
class ChatStreamBusyDetail(BaseModel):
    session_id: str
    message_id: Optional[str] = None


class ChatStreamBusyErrorResponse(BaseModel):
    detail: ChatStreamBusyDetail
