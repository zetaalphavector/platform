"""
Compatibility layer for Pydantic v1 and v2.
Provides a unified BaseModel, GenericModel, and root_validator alias.
"""

from pydantic.version import VERSION as _PYDANTIC_VERSION

PYDANTIC_V2 = _PYDANTIC_VERSION.startswith("2.")

if PYDANTIC_V2:
    from collections import deque
    from decimal import Decimal
    from enum import Enum
    from types import GeneratorType
    from typing import Any, Union

    from pydantic import AwareDatetime  # type: ignore
    from pydantic import BaseModel as _BaseModel  # type: ignore
    from pydantic import (  # type: ignore
        ConfigDict,
        Field,
        GetCoreSchemaHandler,
        GetJsonSchemaHandler,
        PrivateAttr,
        StrictBool,
        StrictFloat,
        StrictInt,
        TypeAdapter,
        field_validator,
        model_validator,
        parse_obj_as,
    )

    # Pydantic v2 imports
    from pydantic.dataclasses import dataclass
    from pydantic.json import pydantic_encoder  # type: ignore
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import ValidationError  # type: ignore
    from pydantic_core import core_schema

    # In v2 BaseModel supports generics natively
    GenericModel = _BaseModel

    class BaseModel(_BaseModel):
        """Extended BaseModel to support v1-style parse methods."""

        def _set_skip_validation(self, name: str, value: Any) -> None:
            """Workaround to be able to set fields without validation."""
            attr = getattr(self.__class__, name, None)
            if isinstance(attr, property):
                attr.__set__(self, value)
            else:
                self.__dict__[name] = value
                self.__pydantic_fields_set__.add(name)

        @classmethod
        def parse_obj(cls, obj):  # type: ignore[override]
            # v1: parse_obj, v2: model_validate
            return cls.model_validate(obj)

        @classmethod
        def parse_raw(cls, raw, **kwargs):  # type: ignore[override]
            # v1: parse_raw, v2: model_validate_json
            return cls.model_validate_json(raw, **kwargs)

        @classmethod
        def from_orm(cls, obj, **kwargs):  # type: ignore[override]
            # v2: use model_validate with from_attributes for ORM
            return cls.model_validate(obj, from_attributes=True)

        @classmethod
        def validate(cls, *args, **kwargs):  # type: ignore[override]
            """Alias for Pydantic v2 validate, forwarding to model_validate."""
            return cls.model_validate(*args, **kwargs)

    def root_validator(*args, **kwargs):  # pylint: disable=unused-argument
        """
        Alias for Pydantic v2 model_validator, mapping v1 pre flag.
        pre=True -> mode='before'. skip_on_failure and allow_reuse are ignored.
        """
        pre = kwargs.pop("pre", False)
        kwargs.pop("skip_on_failure", None)
        kwargs.pop("allow_reuse", None)
        mode = "before" if pre else "after"
        return model_validator(mode=mode, **kwargs)

    def validator(*fields, **kwargs):  # pylint: disable=unused-argument
        """Alias for Pydantic v2 field_validator, wrapping v1-style signature."""

        def decorator(fn):
            base_fn = fn.__func__ if isinstance(fn, classmethod) else fn
            # inspect argument count once
            n_args = base_fn.__code__.co_argcount
            expects_values = n_args >= 3

            def wrapper(cls, v, info):
                values_dict = getattr(info, "data", {}) or {}
                if expects_values:
                    return base_fn(cls, v, values_dict)
                return base_fn(cls, v)

            # Map v1 'pre' and 'always' flags to v2 mode/check_fields
            pre = kwargs.pop("pre", False)
            always = kwargs.pop("always", False)
            mode = "before" if pre else "after"
            # in v2, check_fields=False forces validator even if field missing (v1 always=True)
            return field_validator(
                *fields, mode=mode, check_fields=not always, **kwargs
            )(wrapper)

        return decorator

    def sequence_like(v: Any) -> bool:
        return isinstance(v, (list, tuple, set, frozenset, GeneratorType, deque))

    def str_validator(v: Any) -> Union[str]:
        if isinstance(v, str):
            if isinstance(v, Enum):
                return v.value
            else:
                return v
        elif isinstance(v, (float, int, Decimal)):
            return str(v)
        elif isinstance(v, (bytes, bytearray)):
            return v.decode()
        else:
            raise ValueError("str type expected")

    update_not_none: Any = None  # type: ignore # no equivalent in v2
    constr_length_validator = None  # type: ignore # no equivalent in v2
    create_model_from_typeddict: Any = None  # type: ignore # no equivalent in v2

    __all__ = [
        "PYDANTIC_V2",
        "BaseModel",
        "GenericModel",
        "model_validator",
        "root_validator",
        "sequence_like",
        "Field",
        "pydantic_encoder",
        "ConfigDict",
        "ValidationError",
        "field_validator",
        "validator",
        "StrictBool",
        "StrictFloat",
        "StrictInt",
        "GetCoreSchemaHandler",
        "GetJsonSchemaHandler",
        "JsonSchemaValue",
        "core_schema",
        "AwareDatetime",
        "TypeAdapter",
        "PrivateAttr",
        "dataclass",
    ]
else:
    from datetime import datetime
    from typing import Any

    try:
        from pydantic import AwareDatetime as _AwareDatetime
        from pydantic import TypeAdapter

        parse_datetime = TypeAdapter(_AwareDatetime).validate_python
    except ImportError:
        from pydantic.datetime_parse import parse_datetime  # type: ignore

    # Pydantic v1 imports
    try:
        from pydantic.v1 import BaseModel as _BaseModel  # type: ignore
        from pydantic.v1 import (  # type: ignore
            Field,
            PrivateAttr,
            StrictBool,
            StrictFloat,
            StrictInt,
            ValidationError,
            create_model_from_typeddict,
            parse_obj_as,
            root_validator,
            validator,
        )
        from pydantic.v1.dataclasses import dataclass  # type: ignore
        from pydantic.v1.json import pydantic_encoder  # type: ignore
        from pydantic.v1.utils import sequence_like, update_not_none  # type: ignore
        from pydantic.v1.validators import (  # type: ignore
            constr_length_validator,
            str_validator,
        )
    except ImportError:
        from pydantic.dataclasses import dataclass  # type: ignore
        from pydantic import BaseModel as _BaseModel  # type: ignore
        from pydantic.utils import update_not_none  # type: ignore
        from pydantic.validators import constr_length_validator, str_validator  # type: ignore # noqa
        from pydantic.utils import sequence_like  # type: ignore
        from pydantic import Field, root_validator, validator  # type: ignore
        from pydantic.json import pydantic_encoder  # type: ignore
        from pydantic import ValidationError, parse_obj_as  # type: ignore
        from pydantic import (  # type: ignore
            StrictBool,
            StrictFloat,
            StrictInt,
            create_model_from_typeddict,
            PrivateAttr,
        )

    from pydantic.generics import GenericModel  # type: ignore

    BaseModel = _BaseModel  # type: ignore
    model_validator = None  # type: ignore # no equivalent in v1
    field_validator = None  # type: ignore # no equivalent in v1
    ConfigDict: Any = None  # type: ignore # no equivalent in v1
    GetCoreSchemaHandler: Any = None  # type: ignore # no equivalent in v1
    GetJsonSchemaHandler: Any = None  # type: ignore # no equivalent in v1
    JsonSchemaValue: Any = None  # type: ignore # no equivalent in v1
    core_schema: Any = None  # type: ignore # no equivalent in v1
    TypeAdapter: Any = None  # type: ignore # no equivalent in v1

    class AwareDatetime(datetime):  # type: ignore
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, value):
            value = parse_datetime(value)
            if value.tzinfo is None:
                raise ValueError("Datetime must be timezone-aware")
            return value

    __all__ = [
        "PYDANTIC_V2",
        "BaseModel",
        "GenericModel",
        "root_validator",
        "sequence_like",
        "update_not_none",
        "constr_length_validator",
        "str_validator",
        "Field",
        "pydantic_encoder",
        "ValidationError",
        "parse_obj_as",
        "validator",
        "StrictBool",
        "StrictFloat",
        "StrictInt",
        "GetCoreSchemaHandler",
        "GetJsonSchemaHandler",
        "JsonSchemaValue",
        "core_schema",
        "AwareDatetime",
        "create_model_from_typeddict",
        "PrivateAttr",
        "dataclass",
    ]
