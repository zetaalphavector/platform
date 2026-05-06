from typing import Any, Dict, List, Optional, Union


class FilterTranslator:

    @staticmethod
    def to_filters_configuration(
        simple_filters: Dict[str, Union[str, List[str]]],
    ) -> Dict[str, Any]:
        operands: List[Dict[str, Any]] = []
        for field_path, value in simple_filters.items():
            if isinstance(value, list):
                operands.append(
                    {"is_in": {"field_path": field_path, "field_values": value}}
                )
            else:
                operands.append(
                    {"equals_to": {"field_path": field_path, "field_value": value}}
                )

        if len(operands) == 1:
            return operands[0]
        return {"and_operator": operands}

    @staticmethod
    def merge(*filter_dicts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        non_none = [f for f in filter_dicts if f]
        if not non_none:
            return None
        if len(non_none) == 1:
            return non_none[0]
        return {"and_operator": non_none}
