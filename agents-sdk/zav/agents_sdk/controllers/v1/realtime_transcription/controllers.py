from contextlib import suppress
from typing import AsyncIterator, Literal, Union

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from zav.api.dependencies import get_message_bus
from zav.message_bus import MessageBus

from zav.agents_sdk.controllers.v1.common import get_headers
from zav.agents_sdk.domain import RequestHeaders
from zav.agents_sdk.domain.realtime_transcription import (
    RealtimeTranscriptionNotConfigured,
    error_event,
)
from zav.agents_sdk.handlers import commands

# Speaks the OpenAI Realtime protocol: a client built with
# base_url=<host>/chat/ws/openai/v1 connects unchanged. Mounting is a host
# decision, like mcp_server_router, so hosts can add their own guards.
realtime_transcription_router = APIRouter()


async def _browser_frames(websocket: WebSocket) -> AsyncIterator[Union[str, bytes]]:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        if message.get("text") is not None:
            yield message["text"]
        elif message.get("bytes") is not None:
            yield message["bytes"]


async def _close_with_error(websocket: WebSocket) -> None:
    if websocket.client_state != WebSocketState.CONNECTED:
        return
    with suppress(WebSocketDisconnect):
        await websocket.send_text(
            error_event(
                "server_error",
                "transcription_unavailable",
                "The transcription provider is unavailable",
            )
        )
        await websocket.close(code=1011)


@realtime_transcription_router.websocket(
    "/chat/ws/openai/v1/realtime", name="realtime_transcription"
)
async def realtime_transcription(
    websocket: WebSocket,
    tenant: str = Query(...),
    intent: Literal["transcription"] = Query(...),
    request_headers: RequestHeaders = Depends(get_headers),
    message_bus: MessageBus = Depends(get_message_bus),
):
    await websocket.accept()
    frames = message_bus.handle_stream(
        commands.StreamRealtimeTranscription(
            tenant=tenant,
            browser_frames=_browser_frames(websocket),
            requester_uuid=request_headers.requester_uuid,
        )
    )
    try:
        async for frame in frames:
            await websocket.send_text(str(frame))
    except RealtimeTranscriptionNotConfigured:
        await websocket.close(code=4404)
        return
    except WebSocketDisconnect:
        return
    except Exception:
        # The provider failed after accept (bad key, 401/429, unreachable). The
        # message bus has logged it; the browser gets a reason and 1011.
        await _close_with_error(websocket)
        return
    finally:
        await frames.aclose()
    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.close()
