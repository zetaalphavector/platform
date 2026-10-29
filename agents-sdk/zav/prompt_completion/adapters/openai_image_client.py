import base64
import math
from typing import Any, Dict, Optional, Tuple, Union

import openai
from openai import APIStatusError
from openai import RateLimitError as OpenAiRateLimitError
from openai.types import ImagesResponse
from zav.llm_domain import (
    ImageOutputFormat,
    ImageSettings,
    LLMModelConfiguration,
    LLMModelType,
    LLMProviderName,
    OpenAIConfiguration,
)
from zav.llm_tracing import Span
from zav.logging import logger

from zav.prompt_completion.adapters.errors import (
    generate_content_filter_error,
    generate_rate_limit_error,
)
from zav.prompt_completion.adapters.openai_clients import build_client
from zav.prompt_completion.adapters.tracing import (
    create_image_generation_span,
    end_span,
)
from zav.prompt_completion.client import (
    ImageGenerationClient,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageUsage,
)
from zav.prompt_completion.client_factories import ImageClientFactory

# Sides must be multiples of 16, ratio 1:3 to 3:1; cost scales with pixels, so 1 MP.
_PIXEL_BUDGET = 1024 * 1024
_SIDE_MULTIPLE = 16
_MAX_RATIO = 3.0
_DEFAULT_OUTPUT_FORMAT = ImageOutputFormat.PNG.value


def _parse_size(size: Optional[str]) -> Optional[Tuple[int, int]]:
    width, separator, height = (size or "").partition("x")
    if not separator or not width.isdigit() or not height.isdigit():
        return None
    return int(width), int(height)


def _size_for(aspect_ratio: Tuple[int, int]) -> Tuple[int, int]:
    width, height = aspect_ratio
    ratio = min(max(width / height, 1 / _MAX_RATIO), _MAX_RATIO)
    exact_width = math.sqrt(_PIXEL_BUDGET * ratio)
    return (
        round(exact_width / _SIDE_MULTIPLE) * _SIDE_MULTIPLE,
        round(exact_width / ratio / _SIDE_MULTIPLE) * _SIDE_MULTIPLE,
    )


# OpenAI: 400 moderation_blocked/content_policy_violation; Azure: 200 + contentFilter.
_SAFETY_ERROR_CODES = frozenset(
    {
        "moderation_blocked",
        "content_policy_violation",
        "content_filter",
        "contentfilter",
    }
)
_SAFETY_HINTS = ("safety system", "content policy", "content filter")


def _is_safety_block(code: Optional[str], message: Optional[str]) -> bool:
    if code and code.lower() in _SAFETY_ERROR_CODES:
        return True
    lowered = (message or "").lower()
    return any(hint in lowered for hint in _SAFETY_HINTS)


def _error_message(error: APIStatusError) -> str:
    # The SDK message embeds the whole body; the body holds the provider's sentence.
    body = error.body if isinstance(error.body, dict) else {}
    return str(body.get("message") or error.message)


def _moderation_details(body: Any) -> str:
    error = body.get("error", body) if isinstance(body, dict) else {}
    details = error.get("moderation_details") if isinstance(error, dict) else None
    if not isinstance(details, dict):
        return ""
    stage = details.get("stage")
    categories = details.get("categories") or []
    parts = []
    if stage:
        parts.append(f"stage: {stage}")
    if categories:
        parts.append(f"categories: {', '.join(str(c) for c in categories)}")
    return f" ({'; '.join(parts)})" if parts else ""


@ImageClientFactory.register(LLMProviderName.OPENAI, LLMModelType.IMAGE)
class OpenAiImageClient(ImageGenerationClient):
    def __init__(
        self,
        client: Union[openai.AsyncAzureOpenAI, openai.AsyncOpenAI],
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ):
        self.__client = client
        self.__model_configuration = model_configuration
        self.__span = span

    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: OpenAIConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "OpenAiImageClient":
        client = build_client(vendor_configuration)
        return cls(client=client, model_configuration=model_configuration, span=span)

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        params = self.__request_params(request)
        generation_span = create_image_generation_span(
            prompt=request.prompt,
            model_name=self.__model_configuration.name,
            model_parameters={
                key: value
                for key, value in params.items()
                if key not in ("model", "prompt", "user")
            },
            span=self.__span,
        )
        # SDK retries bill again and triple the wait, so the caller's timeout is final.
        options: Dict[str, Any] = {"max_retries": 0}
        if request.timeout_seconds is not None:
            options["timeout"] = request.timeout_seconds
        client = self.__client.with_options(**options)
        try:
            response = await client.images.generate(**params)
        except OpenAiRateLimitError as e:
            message = _error_message(e)
            if _is_safety_block(e.code, message):
                error: Exception = generate_content_filter_error(
                    f"{message}{_moderation_details(e.body)}"
                )
            else:
                logger.warning(f"Image generation rate limit reached: {e}")
                error = generate_rate_limit_error(message, e.response.headers)
            self.__end_span_with_error(generation_span, str(error))
            raise error from e
        except APIStatusError as e:
            message = _error_message(e)
            if e.status_code < 500 and _is_safety_block(e.code, message):
                error = generate_content_filter_error(
                    f"{message}{_moderation_details(e.body)}"
                )
                self.__end_span_with_error(generation_span, str(error))
                raise error from e
            self.__end_span_with_error(generation_span, message)
            raise
        except Exception as e:
            self.__end_span_with_error(generation_span, str(e))
            raise

        try:
            result = self.__parse_response(
                response,
                params.get("output_format", _DEFAULT_OUTPUT_FORMAT),
                _size_for(request.aspect_ratio),
            )
        except Exception as e:
            self.__end_span_with_error(generation_span, str(e))
            raise
        end_span(
            usage=(
                {
                    "usage": {
                        "input": result.usage.input_tokens,
                        "output": result.usage.output_tokens,
                        "total": result.usage.total_tokens,
                        "unit": "TOKENS",
                    }
                }
                if result.usage
                else {}
            ),
            span=generation_span,
            content=result.revised_prompt,
            model=self.__model_configuration.name,
        )
        return result

    def __request_params(self, request: ImageGenerationRequest) -> Dict[str, Any]:
        configuration = self.__model_configuration
        settings = configuration.image_settings or ImageSettings()
        output_format = settings.output_format
        if (
            request.background == "transparent"
            and output_format == ImageOutputFormat.JPEG
        ):
            # jpeg has no alpha channel.
            output_format = ImageOutputFormat.PNG
        params: Dict[str, Any] = {
            "model": configuration.name,
            "prompt": request.prompt,
            "n": 1,
            "size": "{}x{}".format(*_size_for(request.aspect_ratio)),
        }
        if request.background is not None:
            params["background"] = request.background
        if settings.quality is not None:
            params["quality"] = settings.quality.value
        if output_format is not None:
            params["output_format"] = output_format.value
        if settings.moderation is not None:
            params["moderation"] = settings.moderation.value
        if (
            output_format in (ImageOutputFormat.JPEG, ImageOutputFormat.WEBP)
            and settings.output_compression is not None
        ):
            params["output_compression"] = settings.output_compression
        if request.user:
            params["user"] = request.user
        return params

    def __parse_response(
        self,
        response: ImagesResponse,
        requested_format: str,
        requested_size: Tuple[int, int],
    ) -> ImageGenerationResult:
        error = self.__response_error(response)
        if error is not None:
            code = error.get("code")
            message = error.get("message") or "Image generation failed"
            if _is_safety_block(code, message):
                raise generate_content_filter_error(
                    f"{message}{_moderation_details({'error': error})}"
                )
            raise RuntimeError(f"Image generation failed: {code}: {message}")
        image = response.data[0] if response.data else None
        if image is None or not image.b64_json:
            raise RuntimeError("Image generation returned no image data")
        output_format = response.output_format or requested_format
        usage = (
            ImageUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
            )
            if response.usage
            else None
        )
        width, height = _parse_size(response.size) or requested_size
        return ImageGenerationResult(
            image_bytes=base64.b64decode(image.b64_json),
            mime_type=f"image/{output_format}",
            width=width,
            height=height,
            revised_prompt=image.revised_prompt,
            usage=usage,
        )

    @staticmethod
    def __response_error(response: ImagesResponse) -> Optional[Dict[str, Any]]:
        # Fields outside the SDK schema, such as Azure's error, land in model_extra.
        extra = getattr(response, "model_extra", None) or {}
        error = extra.get("error") if isinstance(extra, dict) else None
        if error is None:
            error = getattr(response, "error", None)
        if isinstance(error, dict):
            return error
        if error is not None:
            return {"code": getattr(error, "code", None), "message": str(error)}
        return None

    @staticmethod
    def __end_span_with_error(span: Optional[Span], message: str) -> None:
        if span:
            span.end(attributes={"level": "ERROR", "status_message": message})


__all__ = ["OpenAiImageClient"]
