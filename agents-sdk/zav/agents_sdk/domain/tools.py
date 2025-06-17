import inspect
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin

from zav.pydantic_compat import BaseModel


def _issubclass_safe(cls, classinfo):
    """Check if a type is subclass of another, handling non-type inputs safely."""
    return isinstance(cls, type) and issubclass(cls, classinfo)


def _get_pydantic_model_schema(model: BaseModel):
    """Convert a Pydantic model to JSON Schema."""
    return model.schema()


def _get_json_type(typ):
    """Translate Python types to JSON Schema types"""
    origin = get_origin(typ)
    # Handle Pydantic models (any class with __fields__ and schema())
    if isinstance(typ, type) and hasattr(typ, "__fields__") and hasattr(typ, "schema"):
        return _get_pydantic_model_schema(typ)
    elif origin is Union:
        union_args = get_args(typ)
        # Filtering out NoneType and considering it as 'Optional'
        non_none_types = [t for t in union_args if t is not type(None)]  # noqa E721
        if len(non_none_types) == 1:
            # It’s essentially an Optional type
            return _get_json_type(non_none_types[0])
        else:
            # It’s a true union, represented as an array of types
            return {"oneOf": [_get_json_type(t) for t in non_none_types]}
    elif origin is list or typ == list:
        type_args = get_args(typ)
        if type_args:
            item_type = type_args[0]
            return {"type": "array", "items": _get_json_type(item_type)}
        else:
            return {"type": "array"}
    elif origin is dict or typ == dict:
        type_args = get_args(typ)
        if type_args:
            _, value_type = type_args
            return {
                "type": "object",
                "additionalProperties": _get_json_type(value_type),
            }
        else:
            return {"type": "object"}
    if typ == str:
        return {"type": "string"}
    elif typ == int:
        return {"type": "integer"}
    elif typ == bool:
        return {"type": "boolean"}
    else:
        return {"type": "string"}  # Default type


class Tool(BaseModel):
    name: str
    description: str
    executable: Callable
    parameters_spec: Optional[Dict[str, Any]] = None

    def get_parameters_spec(self) -> Dict[str, Any]:
        """Returns a JSON schema of the parameters of the tool."""
        if self.parameters_spec:
            return self.parameters_spec
        # If schema is not provided, generate it from the function signature
        # and annotations.
        schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

        signature = inspect.signature(self.executable)
        for name, param in signature.parameters.items():
            param_schema = {}
            param_type = param.annotation

            if param_type is not inspect.Parameter.empty:
                param_schema = _get_json_type(param_type)
            # Add to schema properties
            schema["properties"][name] = param_schema

            # Add to required list if no default value
            if param.default == inspect.Parameter.empty and not (
                get_origin(param_type) == Union and type(None) in get_args(param_type)
            ):
                schema["required"].append(name)

        return schema


def _parse_params(signature: inspect.Signature, params: Optional[Dict[str, Any]]):
    exec_params = dict(params) if params else {}

    for param_name, param in signature.parameters.items():
        if param_name in exec_params and hasattr(param.annotation, "__fields__"):
            exec_params[param_name] = param.annotation(**exec_params[param_name])

    return exec_params


class ToolsRegistry:
    def __init__(self):
        self.tools_index: Dict[str, Tool] = {}

    def extend(self, tools: List[Tool]):
        for tool in tools:
            self.tools_index[tool.name] = tool

    def add(
        self,
        executable: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        qualified_name = (name or executable.__qualname__).replace(".", "_")
        description = description or inspect.getdoc(executable) or ""
        self.tools_index.update(
            {
                qualified_name: Tool(
                    name=qualified_name,
                    description=description,
                    executable=executable,
                )
            }
        )

    async def execute(self, name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if name not in self.tools_index:
            raise ValueError(
                f"Tool {name} not found. Please provide a valid tool name."
            )

        try:
            executable = self.tools_index[name].executable
            # Inspect the executable's parameters
            sig = inspect.signature(executable)
            exec_params = _parse_params(signature=sig, params=params)
            if inspect.iscoroutinefunction(executable):
                exec_response = await executable(**exec_params)  # type: ignore
            else:
                exec_response = executable(**exec_params)

            return exec_response
        except Exception as e:
            raise Exception(f"Error in executing tool {name}: {e}")
