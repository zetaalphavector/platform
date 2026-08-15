from typing import Set, Type

from pydantic import BaseModel, ConfigDict


def prune_hidden_fields(schema: dict, _model: Type[BaseModel]) -> None:
    """``json_schema_extra`` callback that shapes a model's OpenAPI schema:

    * fields whose ``json_schema_extra`` marks them ``hidden`` are dropped from
      the generated schema entirely (and from ``required``), so a model can carry
      internal-only fields — e.g. the raw access-right tokens behind a sharing
      policy — that never surface in the client contract;
    * fields marked ``spec_required`` are promoted into ``required`` (and the
      marker is stripped), so a server-computed field stays required in the
      generated client even though the model gives it a default.

    Mutates ``schema`` in place. Pure schema mechanics, no domain coupling — this
    is the single shared implementation; resource models layer their own domain
    semantics (e.g. sharing) on top.
    """
    props = {}
    hidden_fields: Set[str] = set()
    extra_required: Set[str] = set()
    for k, v in schema.get("properties", {}).items():
        if v.get("spec_required", False):
            extra_required.add(k)
            del v["spec_required"]
        if not v.get("hidden", False):
            props[k] = v
        else:
            hidden_fields.add(k)
    if extra_required:
        schema["required"] = schema.get("required", []) + list(extra_required)
    required = schema.get("required")
    if required and hidden_fields:
        schema["required"] = [r for r in required if r not in hidden_fields]
    schema["properties"] = props


class BaseModelWithHidden(BaseModel):
    """``BaseModel`` whose ``hidden`` fields are pruned from the OpenAPI schema and
    whose ``spec_required`` fields stay required in the generated client. See
    :func:`prune_hidden_fields`."""

    model_config = ConfigDict(
        from_attributes=True, json_schema_extra=prune_hidden_fields
    )
