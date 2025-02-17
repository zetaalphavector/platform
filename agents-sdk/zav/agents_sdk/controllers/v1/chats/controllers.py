from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, Query
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from zav.api.dependencies import get_message_bus
from zav.api.errors import UnknownException
from zav.logging import logger
from zav.message_bus import MessageBus

from zav.agents_sdk.controllers.v1.chats.types import (
    ChatResponseForm,
    ChatResponseItem,
    ChatStreamItem,
)
from zav.agents_sdk.controllers.v1.common import get_headers
from zav.agents_sdk.domain import ChatRequest, RequestHeaders
from zav.agents_sdk.domain.chat_agent import ChatMessage
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


def extract_create_stream_command(
    body: ChatResponseForm,
    tenant: str = Query(...),
    index_id: Optional[str] = Query(None),
    request_headers: RequestHeaders = Depends(get_headers),
) -> commands.CreateChatStream:
    return commands.CreateChatStream(
        tenant=tenant,
        index_id=index_id,
        request_headers=request_headers,
        chat_request=ChatRequest(**body.dict(exclude_unset=True)),
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


async def stream_response(chat_message_stream: AsyncGenerator[ChatMessage, None]):
    if not isinstance(chat_message_stream, AsyncGenerator):
        raise UnknownException("Could not create chat response.")
    try:
        async for message in chat_message_stream:
            yield ServerSentEvent(
                data=message.json(), event="new_message", id="message_id", retry=15000
            )
    except Exception as e:
        logger.error(f"Error in streaming chat messages: {e}", exc_info=True)
        yield ServerSentEvent(data=str(e), event="error")
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
    command=Depends(extract_create_stream_command),
    message_bus: MessageBus = Depends(get_message_bus),
):
    results = await message_bus.handle(command)
    result = results.pop(0)

    if not result:
        raise UnknownException("Could not create chat response.")

    return EventSourceResponse(stream_response(result), media_type="text/event-stream")
