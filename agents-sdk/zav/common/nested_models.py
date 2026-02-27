from typing import Any, Optional

from pydantic import BaseModel

_sentinel = object()


class FieldMapping(BaseModel):
    source_field_name: str
    target_field_name: str
    inner_field_mappings: Optional[list["FieldMapping"]] = None


def getpathattr(obj: object, path: str, default=None) -> Any:
    """A getpathattr function that deals with DCMI fields"""
    if obj == default:
        return default
    if path == ".":
        return obj
    # TODO: fix this by assuming DCMI.xx.yy is escaped as "DCMI.xx.yy"
    path_parts = path.replace("DCMI.", "DCMI__dot__").split(".")
    head_path = path_parts[0].replace("DCMI__dot__", "DCMI.")
    if not head_path:
        return default
    tail_path = (
        ".".join(path_parts[1:]).replace("DCMI__dot__", "DCMI.")
        if len(path_parts) > 1
        else None
    )
    val = _sentinel
    if isinstance(obj, dict):
        val = obj.get(int(head_path) if head_path.isdigit() else head_path, _sentinel)
    elif isinstance(obj, list):
        val = [
            (
                default
                if (v := getpathattr(obj_item, head_path, default=_sentinel))
                == _sentinel
                else v
            )
            for obj_item in obj
        ]
        val = [v for v in val if v != _sentinel]
    else:
        val = getattr(obj, head_path, _sentinel)
        if val == _sentinel:
            # It it starts with custom_metadata means it has attempted to find
            # it under that key already
            if not path.startswith("custom_metadata."):
                return getpathattr(obj, f"custom_metadata.{path}", default=default)

    if val == _sentinel:
        return default
    if tail_path is not None:
        return getpathattr(val, tail_path, default=default)
    return val


def create_jsonpath_dict(json_path: str, value: Any) -> dict:
    result: dict = {}
    for idx, json_path_part in enumerate(
        reversed(json_path.replace("DCMI.", "DCMI__dot__").split("."))
    ):
        result = {
            json_path_part.replace("DCMI__dot__", "DCMI."): (
                value if idx == 0 else result
            )
        }
    return result


def update_nested_dict(orig: dict, update: dict) -> dict:
    for k, v in update.items():
        orig_value = orig.get(k, {}) or {}
        if isinstance(v, dict) and isinstance(orig_value, dict):
            orig[k] = update_nested_dict(orig_value, v)
        else:
            orig[k] = v
    return orig


def update_jsonpath_dict(orig_dict: dict, json_path: str, value: Any):
    update_value = create_jsonpath_dict(json_path=json_path, value=value)
    update_nested_dict(orig=orig_dict, update=update_value)
