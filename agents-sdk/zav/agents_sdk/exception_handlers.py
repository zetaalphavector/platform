from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from zav.agents_sdk.adapters.stream_buffer.resumable_chat_streams import (
    ChatStreamBusyError,
    ChatStreamGoneError,
)
from zav.agents_sdk.domain.chat_conversation_store import ConversationNotOwnedError


def add_exception_handlers(app: FastAPI):
    @app.exception_handler(ValueError)
    async def value_error_request_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ConversationNotOwnedError)
    async def conversation_not_owned_handler(
        request: Request, exc: ConversationNotOwnedError
    ):
        # A turn tried to write a conversation owned by someone else. Sharing is
        # read-only, so this is an authorization failure, not a 500.
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(ChatStreamBusyError)
    async def chat_stream_busy_handler(request: Request, exc: ChatStreamBusyError):
        # One in-flight turn per conversation. Return the occupant so the
        # client can attach to the running stream instead of racing it.
        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "session_id": exc.session_id,
                    "message_id": exc.running_message_id,
                }
            },
        )

    @app.exception_handler(ChatStreamGoneError)
    async def chat_stream_gone_handler(request: Request, exc: ChatStreamGoneError):
        # The conversation's slot moved past this turn: it finished and was
        # overwritten or reaped. 410, not 404 — the client must refetch the
        # conversation, never re-drive a turn that already completed.
        return JSONResponse(status_code=410, content={"detail": str(exc)})
