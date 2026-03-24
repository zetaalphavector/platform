import json
from functools import wraps

from zav.api.errors import UnknownException
from zav.logging import logger


def _extract_error_message(e: Exception) -> str:
    body = getattr(e, "body", None)
    if body:
        parsed = json.loads(body)
        if "message" in parsed:
            return parsed["message"]
        if "detail" in parsed:
            return parsed["detail"]
    return "Server error"


def handle_api_errors(f):
    @wraps(f)
    async def decorated(*args, _retries=0, **kwargs):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            if hasattr(e, "body"):
                msg = _extract_error_message(e)
                logger.exception("API call failed: %s", msg)
                raise UnknownException(msg)
            raise

    return decorated
