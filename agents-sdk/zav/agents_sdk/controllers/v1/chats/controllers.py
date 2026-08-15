import json
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse
from zav.api.dependencies import get_message_bus
from zav.api.error_responses import ApiError, error_responses
from zav.api.errors import BadRequestException, NotFoundException, UnknownException
from zav.logging import logger
from zav.message_bus import MessageBus
from zav.pydantic_compat import PYDANTIC_V2

from zav.agents_sdk.adapters.stream_buffer import (
    StreamBufferLimitError,
    StreamBufferRegistry,
)
from zav.agents_sdk.adapters.stream_buffer.resumable_chat_streams import (
    ResumableChatStreams,
    StreamSource,
)
from zav.agents_sdk.controllers.v1.chats.types import (
    ChatResponseForm,
    ChatResponseItem,
    ChatStreamBusyErrorResponse,
    ChatStreamItem,
    ChatStreamStatusItem,
)
from zav.agents_sdk.controllers.v1.common import get_headers
from zav.agents_sdk.dependencies import (
    get_chat_stream_recording_store,
    get_chat_stream_supervision_store,
    get_stream_buffer_registry,
)
from zav.agents_sdk.domain import ChatRequest, RequestHeaders
from zav.agents_sdk.domain.chat_agent_state_store import (
    AgentsStorageNotConfiguredError,
)
from zav.agents_sdk.domain.chat_stream_recording_store import ChatStreamRecordingStore
from zav.agents_sdk.domain.chat_stream_supervision_store import (
    ChatStreamSupervisionStore,
)
from zav.agents_sdk.handlers import commands

chat_router = APIRouter(tags=["chat"])


def get_resumable_chat_streams(
    message_bus: MessageBus = Depends(get_message_bus),
    stream_buffer_registry: StreamBufferRegistry = Depends(get_stream_buffer_registry),
    chat_stream_recording_store: Optional[ChatStreamRecordingStore] = Depends(
        get_chat_stream_recording_store
    ),
    chat_stream_supervision_store: Optional[ChatStreamSupervisionStore] = Depends(
        get_chat_stream_supervision_store
    ),
) -> ResumableChatStreams:
    return ResumableChatStreams(
        stream_buffer_registry=stream_buffer_registry,
        message_bus=message_bus,
        recording_store=chat_stream_recording_store,
        supervision_store=chat_stream_supervision_store,
    )


def extract_create_response_command(
    body: ChatResponseForm,
    tenant: str = Query(...),
    index_id: Optional[str] = Query(None),
    resume: bool = Query(False),
    message_id: Optional[str] = Query(None),
    request_headers: RequestHeaders = Depends(get_headers),
) -> commands.CreateChatResponse:
    cmd = commands.CreateChatResponse(
        tenant=tenant,
        index_id=index_id,
        request_headers=request_headers,
        chat_request=ChatRequest(
            **body.dict(exclude_unset=True, exclude={"stateful", "session_id"})
        ),
        stateful=bool(body.stateful),
        resume=resume,
    )
    if body.session_id is not None:
        cmd.session_id = body.session_id
    # The turn id is server-assigned; the query param only addresses an
    # existing turn on resume, keeping the takeover under the same stream and
    # supervision-lease key as the interrupted producer.
    if resume and message_id is not None:
        cmd.message_id = message_id
    return cmd


def extract_create_stream_command(
    body: ChatResponseForm,
    tenant: str = Query(...),
    index_id: Optional[str] = Query(None),
    resume: bool = Query(False),
    message_id: Optional[str] = Query(None),
    request_headers: RequestHeaders = Depends(get_headers),
) -> commands.CreateChatStream:
    cmd = commands.CreateChatStream(
        tenant=tenant,
        index_id=index_id,
        request_headers=request_headers,
        chat_request=ChatRequest(
            **body.dict(exclude_unset=True, exclude={"stateful", "session_id"})
        ),
        stateful=bool(body.stateful),
        resume=resume,
    )
    if body.session_id is not None:
        cmd.session_id = body.session_id
    if resume and message_id is not None:
        cmd.message_id = message_id
    return cmd


@chat_router.post(
    "/chats/responses",
    response_model=ChatResponseItem,
    response_model_exclude_none=True,
    status_code=201,
    name="create_chat_response",
    operation_id="create_chat_response",
)
@chat_router.post(
    "/chat/response",
    response_model=ChatResponseItem,
    response_model_exclude_none=True,
    status_code=201,
    name="create_chat",
    operation_id="create_chat",
)
async def create_chat_response(
    command=Depends(extract_create_response_command),
    message_bus: MessageBus = Depends(get_message_bus),
):
    results = await message_bus.handle(command)
    result = results.pop(0)

    if not result:
        raise UnknownException("Could not create chat response.")

    return ChatResponseItem.from_orm(result)


def stream_error_code(exc: Exception) -> Tuple[str, str]:
    """Classify a producer exception into the ``(code, message)`` the SSE
    ``error`` frame carries. Unknown errors stay generic so no exception text
    leaks to the client."""
    if isinstance(exc, AgentsStorageNotConfiguredError):
        return ("AGENTS_STORAGE_NOT_CONFIGURED", str(exc))
    return ("INTERNAL", "Internal streaming error")


async def stream_response(source: StreamSource, start_index: int = 0):
    try:
        async for event_index, message in source.iter_events(start_index=start_index):
            if PYDANTIC_V2:
                data = message.model_dump_json()
            else:
                data = message.json()
            yield ServerSentEvent(
                data=data,
                event="new_message",
                id=f"{source.message_id}:{event_index}",
                retry=15000,
            )
    except Exception as e:
        code, error_message = stream_error_code(e)
        logger.error(f"Error in streaming chat messages: {e}", exc_info=True)
        yield ServerSentEvent(
            data=json.dumps({"code": code, "message": error_message}),
            event="error",
        )
        return
    # A followed recording ends with a verdict the live-buffer path has no
    # need for (its clean close already means "done"). Surface it as a final
    # control event so a reconnected client can tell a sealed turn from one
    # whose producing pod died and is therefore resumable.
    terminal_reason = source.terminal_reason
    if terminal_reason is not None:
        yield ServerSentEvent(
            data=json.dumps({"reason": terminal_reason}),
            event="stream_status",
        )


CHAT_STREAM_BUSY = ApiError(
    409,
    "The conversation already has a turn in flight; attach to the running "
    "stream instead of starting another.",
    model=ChatStreamBusyErrorResponse,
)
CHAT_STREAM_GONE = ApiError(
    410,
    "The conversation's stream slot moved past this turn; refetch the "
    "conversation instead of re-driving it.",
)


@chat_router.post(
    "/chat/stream",
    response_model=ChatStreamItem,
    response_model_exclude_none=True,
    responses={
        201: {
            "content": {
                "text/event-stream": {
                    "schema": {"$ref": "#/components/schemas/ChatStreamItem"}
                }
            },
            "description": "Chat stream response",
        },
        **error_responses(CHAT_STREAM_BUSY),
    },
    status_code=201,
    name="create_chat_streaming",
    operation_id="create_chat_streaming",
)
async def create_chat_streaming(
    command: commands.CreateChatStream = Depends(extract_create_stream_command),
    resumable_chat_streams: ResumableChatStreams = Depends(get_resumable_chat_streams),
):
    try:
        result = await resumable_chat_streams.create(command=command)
    except StreamBufferLimitError as e:
        raise BadRequestException(str(e))

    return EventSourceResponse(
        stream_response(result), media_type="text/event-stream", status_code=201
    )


@chat_router.get(
    "/chat/stream/status",
    response_model=ChatStreamStatusItem,
    name="get_chat_stream_status",
    operation_id="get_chat_stream_status",
)
async def get_chat_stream_status(
    session_id: str = Query(...),
    tenant: str = Query(...),
    request_headers: RequestHeaders = Depends(get_headers),
    resumable_chat_streams: ResumableChatStreams = Depends(get_resumable_chat_streams),
):
    """Which turn occupies the conversation's stream slot, and its state.

    The discovery hook for a client landing on a conversation: a ``running``
    answer means a turn is being produced right now and can be attached to
    via ``GET /chat/stream/{message_id}``. 404 means there is no slot —
    nothing in flight and nothing recent to replay.
    """
    slot = await resumable_chat_streams.status(
        session_id=session_id,
        tenant=tenant,
        requester_uuid=request_headers.requester_uuid,
    )
    if slot is None:
        raise NotFoundException("No stream slot for this conversation")
    return ChatStreamStatusItem(message_id=slot.message_id, status=slot.status)


@chat_router.get(
    "/chat/stream/{message_id}",
    response_model=ChatStreamItem,
    response_model_exclude_none=True,
    responses={
        200: {
            "content": {
                "text/event-stream": {
                    "schema": {"$ref": "#/components/schemas/ChatStreamItem"}
                }
            },
            "description": "Chat stream response",
        },
        **error_responses(CHAT_STREAM_GONE),
    },
    status_code=200,
    name="get_chat_stream_events",
    operation_id="get_chat_stream_events",
)
async def get_chat_stream_events(
    message_id: str,
    tenant: str = Query(...),
    start_index: int = Query(0, ge=0),
    session_id: Optional[str] = Query(None),
    partial: bool = Query(False),
    request_headers: RequestHeaders = Depends(get_headers),
    resumable_chat_streams: ResumableChatStreams = Depends(get_resumable_chat_streams),
):
    result = await resumable_chat_streams.get(
        message_id=message_id,
        tenant=tenant,
        requester_uuid=request_headers.requester_uuid,
        session_id=session_id,
        partial=partial,
    )
    if result is None:
        raise NotFoundException("Stream not found or expired")

    return EventSourceResponse(
        stream_response(result, start_index=start_index),
        media_type="text/event-stream",
    )


@chat_router.delete(
    "/chat/stream/{message_id}",
    response_class=Response,
    status_code=204,
    name="cancel_chat_stream",
    operation_id="cancel_chat_stream",
)
async def cancel_chat_stream(
    message_id: str,
    tenant: str = Query(...),
    request_headers: RequestHeaders = Depends(get_headers),
    resumable_chat_streams: ResumableChatStreams = Depends(get_resumable_chat_streams),
):
    cancelled = await resumable_chat_streams.cancel(
        message_id=message_id,
        tenant=tenant,
        requester_uuid=request_headers.requester_uuid,
    )
    if not cancelled:
        raise NotFoundException("Stream not found or expired")
    return None
