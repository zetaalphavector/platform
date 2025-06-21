from __future__ import annotations

from typing import Any, Dict, Optional, Set

# Fields that must always be supplied by the service layer / handler itself and
# therefore must not be overridden by user-supplied ``bot_params``.
_PROTECTED_HANDLER_PARAMS: Set[str] = {"tenant", "request_headers", "index_id"}

__all__ = [
    "_PROTECTED_HANDLER_PARAMS",
    "sanitize_bot_params",
]


def sanitize_bot_params(bot_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of *bot_params* with security-critical keys stripped.

    This prevents callers from overriding values like ``tenant`` or
    ``request_headers`` that are controlled by the upstream service layer.
    """

    if not bot_params:
        return {}

    # Remove any keys that clash with protected parameters.
    return {k: v for k, v in bot_params.items() if k not in _PROTECTED_HANDLER_PARAMS}
