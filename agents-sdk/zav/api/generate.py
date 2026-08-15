import argparse
import os
from importlib import import_module
from typing import Dict, List, Tuple

import yaml
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from zav.api.setup_routers import setup_routers


def _remove_additional_properties(mapping: Dict) -> None:
    if mapping.get("additionalProperties") is True:
        mapping.pop("additionalProperties")


def _ensure_enum_description(mapping: Dict) -> None:
    if "enum" in mapping and "description" not in mapping:
        mapping["description"] = "An enumeration."


def _collect_union_items(union_list: List) -> Tuple[List, int]:
    non_null_items: List = []
    null_count = 0
    for item in union_list:
        if isinstance(item, Dict):
            _remove_additional_properties(item)
            if item.get("type") == "null":
                null_count += 1
                continue
        non_null_items.append(item)
    return non_null_items, null_count


def _collapse_nullable_union(mapping: Dict) -> None:
    for union_key in ("anyOf", "oneOf"):
        union_list = mapping.get(union_key)
        if not isinstance(union_list, List):
            continue
        non_null_items, null_count = _collect_union_items(union_list)
        if (
            null_count >= 1
            and len(non_null_items) == 1
            and isinstance(non_null_items[0], Dict)
        ):
            replacement = non_null_items[0]
            original_title = mapping.get("title")
            original_description = mapping.get("description")
            original_default = mapping.get("default")
            deprecated = mapping.get("deprecated")
            mapping.pop(union_key)
            mapping.clear()

            for key, value in replacement.items():
                mapping[key] = value
            if original_title is not None and "title" not in mapping:
                mapping["title"] = original_title
            if original_description is not None and "description" not in mapping:
                mapping["description"] = original_description
            if original_default is not None and "default" not in mapping:
                mapping["default"] = original_default
            if deprecated is not None and "deprecated" not in mapping:
                mapping["deprecated"] = deprecated
        elif null_count >= 1 and len(non_null_items) >= 1:
            mapping[union_key] = non_null_items


def _convert_const_to_enum(mapping: Dict) -> None:
    if "const" in mapping and "enum" not in mapping:
        mapping["enum"] = [mapping.pop("const")]


def _normalize_validation_error(mapping: Dict) -> None:
    # Pydantic v2 adds 'ctx' and 'input' to the FastAPI ValidationError schema.
    # Strip them so the spec stays stable across Pydantic versions.
    required = mapping.get("required")
    if isinstance(required, list) and set(required) == {"loc", "msg", "type"}:
        properties = mapping.get("properties", {})
        properties.pop("ctx", None)
        properties.pop("input", None)


def _normalize_bytes_field(mapping: Dict) -> None:
    if mapping.get("contentMediaType") == "application/octet-stream":
        mapping.pop("contentMediaType")
        mapping.pop("contentEncoding", None)
        mapping["format"] = "binary"


def _prune_nullable_required_fields(mapping: Dict) -> None:
    required = mapping.get("required")
    properties = mapping.get("properties")
    if not isinstance(required, List) or not isinstance(properties, Dict):
        return
    nullable_fields = []
    for field in required:
        prop = properties.get(field)
        if not isinstance(prop, Dict):
            continue
        candidates = prop.get("anyOf")
        if not isinstance(candidates, List):
            continue
        has_null = any(
            isinstance(option, Dict) and option.get("type") == "null"
            for option in candidates
        )
        if has_null:
            nullable_fields.append(field)
    for field in nullable_fields:
        mapping["required"].remove(field)


def _normalize_mapping(mapping: Dict) -> None:
    _remove_additional_properties(mapping)
    _normalize_bytes_field(mapping)
    _normalize_validation_error(mapping)
    _convert_const_to_enum(mapping)
    _collapse_nullable_union(mapping)
    _normalize_bytes_field(mapping)  # re-run: collapse may expose contentMediaType
    _ensure_enum_description(mapping)
    _prune_nullable_required_fields(mapping)
    for child in list(mapping.values()):
        _normalize_schema_inplace(child)


def _normalize_schema_inplace(node: object) -> None:
    if isinstance(node, Dict):
        _normalize_mapping(node)
    elif isinstance(node, List):
        for item in node:
            _normalize_schema_inplace(item)


if __name__ == "__main__":
    app = FastAPI()
    app.openapi_version = "3.0.2"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=str,
        default="./src/openapi/spec.yaml",
        help="Where to store the generated OpenAPI spec.",
    )

    parser.add_argument(
        "--routers-module",
        type=str,
        default="src.controllers",
        help="Module where `routers` are defined.",
    )
    args = parser.parse_args()

    # override app.openapi to post-process the generated schema and normalize
    # problematic fields for the openapi generator

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            servers=app.servers,
            separate_input_output_schemas=False,
            openapi_version="3.0.2",
        )
        # Apply normalization across the entire OpenAPI document tree
        _normalize_schema_inplace(openapi_schema)
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[assignment]

    controllers = import_module(args.routers_module)
    setup_routers(app=app, routers=controllers.routers)

    os.makedirs(os.path.dirname(args.destination), exist_ok=True)
    with open(args.destination, "w") as f:
        yaml.dump(app.openapi(), f)
