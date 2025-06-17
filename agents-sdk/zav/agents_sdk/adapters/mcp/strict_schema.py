# Adapted from
# https://github.com/openai/openai-agents-python/blob/main/src/agents/strict_schema.py
from typing import Any, Dict, Tuple

_sentinel = object()


def ensure_strict_json_schema(
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the OpenAI API expects.
    """
    if not schema:
        return {
            "additionalProperties": False,
            "type": "object",
            "properties": {},
            "required": [],
        }
    return _ensure_strict_json_schema(schema, path=(), root=schema)


def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: Tuple[str, ...],
    root: Dict[str, object],
) -> Dict[str, Any]:
    if not isinstance(json_schema, dict):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    defs = json_schema.get("$defs")
    if isinstance(defs, dict):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(
                def_schema, path=(*path, "$defs", def_name), root=root
            )

    definitions = json_schema.get("definitions")
    if isinstance(definitions, dict):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(
                definition_schema,
                path=(*path, "definitions", definition_name),
                root=root,
            )

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False
    elif (
        typ == "object"
        and "additionalProperties" in json_schema
        and json_schema["additionalProperties"]
    ):
        raise ValueError(
            "additionalProperties should not be set for object types."
            " This could be because you're using an older version of Pydantic, "
            "or because you configured additional properties to be allowed."
            " If you really need this, update the function or output tool "
            "to not use a strict schema."
        )

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if isinstance(properties, dict):
        json_schema["required"] = list(properties.keys())
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(
                prop_schema, path=(*path, "properties", key), root=root
            )
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if isinstance(items, dict):
        json_schema["items"] = _ensure_strict_json_schema(
            items, path=(*path, "items"), root=root
        )

    # unions
    any_of = json_schema.get("anyOf")
    if isinstance(any_of, list):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(
                variant, path=(*path, "anyOf", str(i)), root=root
            )
            for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if isinstance(all_of, list):
        if len(all_of) == 1:
            json_schema.update(
                _ensure_strict_json_schema(
                    all_of[0], path=(*path, "allOf", "0"), root=root
                )
            )
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(
                    entry, path=(*path, "allOf", str(i)), root=root
                )
                for i, entry in enumerate(all_of)
            ]

    # strip `None` defaults as there's no meaningful distinction here
    # the schema will still be `nullable` and the model will default
    # to using `None` anyway
    if json_schema.get("default", _sentinel) is None:
        json_schema.pop("default")

    # we can't use `$ref`s if there are also other properties defined, e.g.
    # `{"$ref": "...", "description": "my description"}`
    #
    # so we unravel the ref
    # `{"type": "string", "description": "my description"}`
    ref = json_schema.get("$ref")
    if ref and has_more_than_n_keys(json_schema, 1):
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = resolve_ref(root=root, ref=ref)
        if not isinstance(resolved, dict):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}"
            )

        # properties from the json schema take priority over the ones on the `$ref`
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        # Since the schema expanded from `$ref` might not have
        # `additionalProperties: false` applied
        # we call `_ensure_strict_json_schema` again to fix the inlined schema
        # and ensure it's valid
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def resolve_ref(*, root: Dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert isinstance(
            value, dict
        ), f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        resolved = value

    return resolved


def has_more_than_n_keys(obj: Dict[str, object], n: int) -> bool:
    i = 0
    for _ in obj.keys():
        i += 1
        if i > n:
            return True
    return False
