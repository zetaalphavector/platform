# ---- Start of patch ----
# Workaround for https://github.com/pydantic/pydantic/issues/5821
#
# Add this code once at the start of your program. It plays nice with mypy.
#
# It patches `typing.Literal` to refer to `typing_extensions.Literal`,
# and thereby avoids a bug in Pydantic<=1.10.7 code for is_literal_type.
import typing

try:
    import typing_extensions

    # Patch Literal in typing to avoid Pydantic<=1.10.7 is_literal_type bug
    typing.Literal = typing_extensions.Literal  # type: ignore
except ImportError:
    # typing_extensions not available, skip Literal patch
    pass
# ---- End of patch ----
# flake8: noqa E402
