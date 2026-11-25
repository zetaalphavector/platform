import email.message
import io
from typing import Any, Callable, Coroutine, Mapping, Optional, Sequence, Type

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ValidationError
from pydantic_core import from_json

from zav.api.intake_admission import get_intake_admission

# FastAPI only checks the body is non-empty before asking the request for its JSON.
VALIDATED_BODY_MARKER = b"{}"


class ValidatedJsonRequest(Request):
    def __init__(self, request: Request, body: bytes, validated: Any) -> None:
        super().__init__(request.scope, request.receive)
        self.__body = body
        self.__validated = validated

    async def body(self) -> bytes:
        return self.__body

    async def json(self) -> Any:
        return self.__validated


def is_json_content_type(content_type: Optional[str]) -> bool:
    if not content_type:
        return False
    message = email.message.Message()
    message["content-type"] = content_type
    subtype = message.get_content_subtype()
    return message.get_content_maintype() == "application" and (
        subtype == "json" or subtype.endswith("+json")
    )


def content_length(request: Request) -> Optional[int]:
    header = request.headers.get("content-length")
    return int(header) if header is not None and header.isdigit() else None


async def read_body(request: Request) -> bytes:
    buffer = io.BytesIO()
    async for chunk in request.stream():
        buffer.write(chunk)
    # getvalue() hands over the buffer without a copy, and the JSON parser reads
    # bytes in place where it copies a bytearray first.
    return buffer.getvalue()


class JsonBodyRoute(APIRoute):
    """Reads a JSON body once the process's intake memory admits it, parses it
    once, validates the body model from the parsed objects and frees the raw body,
    and answers every rejection with a 422 that leaves out the rejected input.
    FastAPI's own parsing keeps the raw body and its parsed JSON on the request
    until the response, and echoes the rejected input back.
    The app binds an IntakeAdmission to `get_intake_admission` in its lifespan.
    A route without a single body model is handled by FastAPI unchanged."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()
        body_model = self.__body_model()
        if body_model is None:
            return handler

        async def validate_then_handle(request: Request) -> Response:
            raw = await read_body(request)
            if not raw:
                return await handler(ValidatedJsonRequest(request, raw, None))
            # Parsed first and validated from the parsed objects, which the model
            # and a validation error share. Validating the bytes directly copies
            # the rejected input into the error: 3x the body, measured 2026-11-15.
            errors: Optional[Sequence[Mapping[str, Any]]] = None
            try:
                parsed = from_json(raw)
            except ValueError as e:
                errors = [
                    {"type": "json_invalid", "loc": (), "msg": f"Invalid JSON: {e}"}
                ]
            del raw
            if errors is None:
                try:
                    validated = body_model.model_validate(parsed)
                except ValidationError as e:
                    errors = e.errors(include_url=False, include_input=False)
                del parsed
            # Raised outside the except block: as the new error's context, the
            # ValidationError would keep the parsed input alive until the response.
            if errors is not None:
                raise RequestValidationError(
                    [{**error, "loc": ("body", *error["loc"])} for error in errors]
                )
            return await handler(
                ValidatedJsonRequest(request, VALIDATED_BODY_MARKER, validated)
            )

        async def admit_then_handle(request: Request) -> Response:
            if not is_json_content_type(request.headers.get("content-type")):
                raise RequestValidationError(
                    [
                        {
                            "type": "model_attributes_type",
                            "loc": ("body",),
                            "msg": "Input should be a JSON object",
                        }
                    ]
                )
            admission = await get_intake_admission(request)
            async with admission.admitted(content_length(request)):
                return await validate_then_handle(request)

        return admit_then_handle

    def __body_model(self) -> Optional[Type[BaseModel]]:
        if self.body_field is None or self._embed_body_fields:
            return None
        annotation = self.body_field.field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
        return None
