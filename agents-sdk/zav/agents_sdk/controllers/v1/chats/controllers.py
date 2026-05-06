from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from zav.api.dependencies import get_message_bus
from zav.api.errors import UnknownException
from zav.logging import logger
from zav.message_bus import MessageBus
from zav.pydantic_compat import PYDANTIC_V2

from zav.agents_sdk.adapters.stream_buffer import StreamBuffer
from zav.agents_sdk.controllers.v1.chats.types import (
    ChatResponseForm,
    ChatResponseItem,
    ChatStreamItem,
)
from zav.agents_sdk.controllers.v1.common import get_headers
from zav.agents_sdk.domain import ChatRequest, RequestHeaders
from zav.agents_sdk.handlers import commands

chat_router = APIRouter(tags=["chat"])


def extract_create_response_command(
    body: ChatResponseForm,
    tenant: str = Query(...),
    index_id: Optional[str] = Query(None),
    request_headers: RequestHeaders = Depends(get_headers),
) -> commands.CreateChatResponse:
    return commands.CreateChatResponse(
        tenant=tenant,
        index_id=index_id,
        request_headers=request_headers,
        chat_request=ChatRequest(**body.dict(exclude_unset=True)),
    )


def extract_create_buffered_stream_command(
    body: ChatResponseForm,
    tenant: str = Query(...),
    index_id: Optional[str] = Query(None),
    request_headers: RequestHeaders = Depends(get_headers),
) -> commands.CreateBufferedChatStream:
    return commands.CreateBufferedChatStream(
        tenant=tenant,
        index_id=index_id,
        request_headers=request_headers,
        chat_request=ChatRequest(**body.dict(exclude_unset=True)),
    )


def extract_get_stream_command(
    message_id: str,
    tenant: str = Query(...),
    request_headers: RequestHeaders = Depends(get_headers),
) -> commands.GetChatStreamBuffer:
    return commands.GetChatStreamBuffer(
        message_id=message_id,
        tenant=tenant,
        requester_uuid=request_headers.requester_uuid,
    )


def extract_cancel_stream_command(
    message_id: str,
    tenant: str = Query(...),
    request_headers: RequestHeaders = Depends(get_headers),
) -> commands.CancelChatStream:
    return commands.CancelChatStream(
        message_id=message_id,
        tenant=tenant,
        requester_uuid=request_headers.requester_uuid,
    )


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


async def stream_response(buffer: StreamBuffer, start_index: int = 0):
    try:
        async for event_index, message in buffer.iter_events(start_index=start_index):
            if PYDANTIC_V2:
                data = message.model_dump_json()
            else:
                data = message.json()
            yield ServerSentEvent(
                data=data,
                event="new_message",
                id=f"{buffer.message_id}:{event_index}",
                retry=15000,
            )
    except Exception as e:
        logger.error(f"Error in streaming chat messages: {e}", exc_info=True)
        yield ServerSentEvent(data="Internal streaming error", event="error")
        return


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
    },
    status_code=201,
    name="create_chat_streaming",
    operation_id="create_chat_streaming",
)
async def create_chat_streaming(
    command=Depends(extract_create_buffered_stream_command),
    message_bus: MessageBus = Depends(get_message_bus),
):
    results = await message_bus.handle(command)
    result = results.pop(0)

    if not result:
        raise UnknownException("Could not create chat response.")

    return EventSourceResponse(stream_response(result), media_type="text/event-stream")


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
    },
    status_code=200,
    name="get_chat_stream_events",
    operation_id="get_chat_stream_events",
)
async def get_chat_stream_events(
    command=Depends(extract_get_stream_command),
    start_index: int = Query(0, ge=0),
    message_bus: MessageBus = Depends(get_message_bus),
):
    results = await message_bus.handle(command)
    result = results.pop(0)

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
    command=Depends(extract_cancel_stream_command),
    message_bus: MessageBus = Depends(get_message_bus),
):
    await message_bus.handle(command)
    return None
