# Workaround for Pydantic v1 + PEP 749 (Python 3.14+) deferred annotations.
#
# PEP 749 defers annotation evaluation — __annotations__ is no longer in the
# namespace passed to type.__new__. Pydantic v1's ModelMetaclass reads
# namespace["__annotations__"] directly, gets nothing, and silently drops all
# annotated fields.
#
# This affects both standalone pydantic v1 and the pydantic.v1 compatibility
# shim in pydantic v2 (used by langfuse and other libraries with Fern-generated
# API types).
#
# Fix: Patches ModelMetaclass.__new__ to eagerly evaluate annotations via
# annotationlib.call_annotate_function() before Pydantic v1 processes them.
#
# USAGE: import this module before any Pydantic v1 model definitions execute.
# It is safe to import multiple times (the patch is idempotent).
#
# REMOVAL TRIGGER: Remove when Pydantic ships PEP 749 support for its v1
# compat layer (https://github.com/pydantic/pydantic/issues/11659), OR when
# we drop all Pydantic v1 model usage from the codebase.
import sys

if sys.version_info >= (3, 14):
    import annotationlib
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="The `dict` method is deprecated; use `model_dump` instead",
        category=DeprecationWarning,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Core Pydantic V1", category=UserWarning
        )
        import pydantic.v1.main as _pydantic_v1_main

    _already_patched = getattr(
        _pydantic_v1_main.ModelMetaclass, "_py314_patched", False
    )

    if not _already_patched:
        _original_model_metaclass_new = _pydantic_v1_main.ModelMetaclass.__new__

        def _patched_model_metaclass_new(mcs, name, bases, namespace, **kwargs):  # type: ignore # noqa: E501
            if "__annotations__" not in namespace:
                annotate_func = namespace.get("__annotate_func__")
                if annotate_func is not None:
                    try:
                        annotations = annotationlib.call_annotate_function(
                            annotate_func,
                            annotationlib.Format.VALUE,
                        )
                        namespace["__annotations__"] = annotations
                    except Exception:
                        pass
            return _original_model_metaclass_new(mcs, name, bases, namespace, **kwargs)

        _pydantic_v1_main.ModelMetaclass.__new__ = _patched_model_metaclass_new  # type: ignore # noqa: E501
        _pydantic_v1_main.ModelMetaclass._py314_patched = True  # type: ignore
