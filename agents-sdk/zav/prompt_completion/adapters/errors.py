import re
from typing import Mapping, Optional

from zav.prompt_completion.client import (
    ContentFilterError,
    PromptTooLargeError,
    RateLimitError,
)


def generate_content_filter_error(message: str) -> ContentFilterError:
    return ContentFilterError(message)


def __extract_int_from_text(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return None


def generate_prompt_too_long_error(error_message: str) -> PromptTooLargeError:
    max_tokens_limit = __extract_int_from_text(
        r"maximum context length is (\d+) tokens", error_message
    )
    requested_tokens = __extract_int_from_text(
        r"resulted in (\d+) tokens", error_message
    )
    extra_tokens = (
        requested_tokens - max_tokens_limit
        if max_tokens_limit and requested_tokens
        else None
    )
    return PromptTooLargeError(error_message, extra_tokens)


_DURATION_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def __parse_go_duration(value: str) -> Optional[float]:
    # OpenAI's x-ratelimit-reset-* headers use Go's duration format: "1s", "6m0s"
    parts = _DURATION_PART.findall(value)
    if not parts or "".join(n + u for n, u in parts) != value.strip():
        return None
    return sum(float(n) * _DURATION_UNIT_SECONDS[u] for n, u in parts)


def __parse_retry_after_seconds(value: str) -> Optional[float]:
    # RFC 7231 allows an HTTP-date here as well; only delta-seconds is honoured
    return float(value) if value.strip().isdigit() else None


def generate_rate_limit_error(
    message: str, headers: Optional[Mapping[str, str]] = None
) -> RateLimitError:
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    seconds = next(
        (
            parsed
            for header, parse in (
                ("retry-after", __parse_retry_after_seconds),
                ("x-ratelimit-reset-requests", __parse_go_duration),
                ("x-ratelimit-reset-tokens", __parse_go_duration),
            )
            if header in headers and (parsed := parse(headers[header])) is not None
        ),
        None,
    )
    if seconds is None:
        # Azure states the delay in the message body instead of a header
        body_hint = __extract_int_from_text(
            r"retry after (\d+) seconds?", message.lower()
        )
        seconds = float(body_hint) if body_hint is not None else None
    return RateLimitError(message, retry_after_seconds=seconds)
