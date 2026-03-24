import copy

from zav.common.nested_models import FieldMapping, getpathattr, update_jsonpath_dict

_sentinel = object()


def map_fields(
    source: dict,
    field_mappings: list[FieldMapping],
    keep_original_fields: bool = False,
) -> dict:
    content_values: dict = {}
    mapped_roots = {fm.source_field_name.split(".", 1)[0] for fm in field_mappings}
    for field_mapping in field_mappings:
        source_name = field_mapping.source_field_name
        target_name = field_mapping.target_field_name
        inner_field_mappings = field_mapping.inner_field_mappings
        field_value = getpathattr(source, source_name, _sentinel)
        if field_value != _sentinel:
            if inner_field_mappings is not None:
                if isinstance(field_value, list):
                    inner_field_values = []
                    for field_value_item in field_value:
                        if field_value_item == _sentinel:
                            continue
                        _v = map_fields(
                            source=field_value_item,
                            field_mappings=inner_field_mappings,
                            keep_original_fields=keep_original_fields,
                        )
                        if _v:
                            inner_field_values.append(_v)
                    field_value = inner_field_values
                else:
                    field_value = map_fields(
                        source=field_value,
                        field_mappings=inner_field_mappings,
                        keep_original_fields=keep_original_fields,
                    )
            update_jsonpath_dict(
                orig_dict=content_values,
                json_path=target_name,
                value=field_value,
            )

    if keep_original_fields and isinstance(source, dict):
        for key, val in source.items():
            if key not in mapped_roots:
                content_values[key] = copy.deepcopy(val)
    return content_values
