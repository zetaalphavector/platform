import inspect
import types
from typing import Union, get_args, get_origin

from zav.pydantic_compat import BaseModel, _BaseModel


def is_union(obj) -> bool:
    if obj is Union:
        return True
    _union_type = getattr(types, "UnionType", None)
    if _union_type:
        return obj is _union_type or isinstance(obj, _union_type)
    return False


def check_is_optional(field):
    origin = get_origin(field)
    return is_union(origin) and type(None) in get_args(field)


def check_is_class(annotation):
    return inspect.isclass(annotation)


def check_is_base_model(annotation):
    return issubclass(annotation, BaseModel) or issubclass(annotation, _BaseModel)
