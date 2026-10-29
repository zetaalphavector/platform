import math

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from zav.prompt_completion import ContentFilterError, RateLimitError

from zav.agents_sdk.adapters.stream_buffer.resumable_chat_streams import (
    ChatStreamBusyError,
    ChatStreamGoneError,
)
from zav.agents_sdk.domain.chat_conversation_store import ConversationNotOwnedError
from zav.agents_sdk.handlers.chats import (
    ChatMessageAlreadyReadError,
    ChatResponseTimeoutError,
)


def rate_limit_message(exc: RateLimitError) -> str:
    retry_after = (
        f"after {int(math.ceil(exc.retry_after_seconds))} seconds"
        if exc.retry_after_seconds is not None
        else "in a moment"
    )
    return (
        "You have reached the rate limit for this model. Please retry "
        f"{retry_after}. If this issue persists, contact "
        "support@zeta-alpha.com for assistance."
    )


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

    @app.exception_handler(ChatMessageAlreadyReadError)
    async def chat_message_already_read_handler(
        request: Request, exc: ChatMessageAlreadyReadError
    ):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(request: Request, exc: RateLimitError):
        # The model provider throttled the turn. 429 with the provider's own
        # Retry-After so a programmatic caller can back off and retry rather than
        # treating it as a server fault.
        headers = (
            {"Retry-After": str(int(exc.retry_after_seconds))}
            if exc.retry_after_seconds is not None
            else {}
        )
        return JSONResponse(
            status_code=429,
            content={"detail": rate_limit_message(exc)},
            headers=headers,
        )

    @app.exception_handler(ChatResponseTimeoutError)
    async def chat_response_timeout_handler(
        request: Request, exc: ChatResponseTimeoutError
    ):
        # The turn exceeded its caller-supplied budget and was cancelled. 504 so a
        # programmatic caller can distinguish a deadline from a server fault and
        # decide whether to retry, rather than treating it as a 500.
        return JSONResponse(status_code=504, content={"detail": str(exc)})

    @app.exception_handler(ContentFilterError)
    async def content_filter_handler(request: Request, exc: ContentFilterError):
        # The model provider blocked the input under its content/safety policy.
        # Deterministic and terminal, so surface a machine-readable code (422)
        # instead of the catch-all 500 the caller cannot distinguish; a caller
        # must not retry it.
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc), "code": "content_filter"},
        )
