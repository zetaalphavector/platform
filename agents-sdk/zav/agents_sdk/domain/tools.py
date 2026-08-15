import copy
import inspect
from typing import Annotated, Any, Callable, Dict, List, Optional, get_args, get_origin

from jinja2 import Undefined
from jinja2.sandbox import SandboxedEnvironment
from zav.pydantic_compat import PYDANTIC_V2, BaseModel, ConfigDict

from zav.agents_sdk.domain.utils import is_union


class _SilentUndefined(Undefined):
    def __str__(self) -> str:
        return ""

    def __html__(self) -> str:
        return ""

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(())

    def __getattr__(self, _name: str):  # type: ignore[no-untyped-return]
        return self

    def __getitem__(self, _key: Any):  # type: ignore[no-untyped-return]
        return self


class ToolStreamingConfig(BaseModel):
    running_text: str
    completed_text: Optional[str] = None
    params_transform: Optional[Callable[[Optional[Dict]], Optional[Dict]]] = None
    response_transform: Optional[Callable[[Optional[Dict]], Optional[Dict]]] = None
    llm_response_transform: Optional[Callable[[Optional[Dict]], Optional[Dict]]] = None

    if PYDANTIC_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:

        class Config:
            arbitrary_types_allowed = True


def streamable(
    running_text: str,
    completed_text: Optional[str] = None,
    params_transform: Optional[Callable[[Optional[Dict]], Optional[Dict]]] = None,
    response_transform: Optional[Callable[[Optional[Dict]], Optional[Dict]]] = None,
    llm_response_transform: Optional[Callable[[Optional[Dict]], Optional[Dict]]] = None,
):
    def decorator(func: Callable) -> Callable:
        setattr(
            func,
            "_streaming_config",
            ToolStreamingConfig(
                running_text=running_text,
                completed_text=completed_text,
                params_transform=params_transform,
                response_transform=response_transform,
                llm_response_transform=llm_response_transform,
            ),
        )
        return func

    return decorator


def apply_transform(
    data: Optional[Dict],
    transform_fn: Optional[Callable[[Optional[Dict]], Optional[Dict]]] = None,
) -> Optional[Dict]:
    if transform_fn is not None:
        return transform_fn(data)
    return data


def hide(data: Optional[Dict]) -> Optional[Dict]:
    return None


def include_fields(*fields: str) -> Callable[[Optional[Dict]], Optional[Dict]]:
    def transform(data: Optional[Dict]) -> Optional[Dict]:
        if data is None:
            return None
        return {k: v for k, v in data.items() if k in fields}

    return transform


def exclude_fields(*fields: str) -> Callable[[Optional[Dict]], Optional[Dict]]:
    def transform(data: Optional[Dict]) -> Optional[Dict]:
        if data is None:
            return None
        return {k: v for k, v in data.items() if k not in fields}

    return transform


def format_display_text(
    template: str,
    tool_params: Optional[Dict[str, Any]] = None,
    tool_result: Optional[Any] = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> str:
    context: Dict[str, Any] = {**(defaults or {})}
    context.update(tool_params or {})
    if tool_result is not None and isinstance(tool_result, dict):
        context.update(tool_result)

    env = SandboxedEnvironment(undefined=_SilentUndefined, autoescape=False)
    return env.from_string(template).render(**context)


def _issubclass_safe(cls, classinfo):
    """Check if a type is subclass of another, handling non-type inputs safely."""
    return isinstance(cls, type) and issubclass(cls, classinfo)


def _inline_refs(schema: Dict) -> Dict:
    """Replace $ref in the schema with inline definitions."""
    defs = schema.get("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_key = ref_path.split("/")[-1]
                    resolved = copy.deepcopy(defs[def_key])
                    return _resolve(resolved)
            return {k: _resolve(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    resolved_schema = _resolve(schema)
    resolved_schema.pop("$defs", None)
    return resolved_schema


def _get_pydantic_model_schema(model: BaseModel):
    """Convert a Pydantic model to a fully inlined JSON Schema."""
    schema = model.schema()
    # definitions are ignored by LLM providers, so we need to inline them
    return _inline_refs(schema)


def _get_json_type(typ):
    """Translate Python types to JSON Schema types"""
    origin = get_origin(typ)

    # Handle Annotated types
    if origin is Annotated:
        args = get_args(typ)
        if args:
            # First arg is the actual type
            actual_type = args[0]
            # Extract metadata (like FieldInfo)
            metadata = args[1:] if len(args) > 1 else []

            # Get the base schema from the actual type
            schema = _get_json_type(actual_type)

            # Extract description from FieldInfo if present
            for meta in metadata:
                if hasattr(meta, "description") and meta.description:
                    schema["description"] = meta.description
                    break

            return schema

    # Handle Pydantic models (any class with __fields__ and schema())
    if isinstance(typ, type) and hasattr(typ, "__fields__") and hasattr(typ, "schema"):
        return _get_pydantic_model_schema(typ)
    elif is_union(origin):
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
    streaming_config: Optional[ToolStreamingConfig] = None
    max_output_tokens: Optional[int] = None

    @classmethod
    def from_callable(
        cls,
        executable: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        streaming_config: Optional[ToolStreamingConfig] = None,
        max_output_tokens: Optional[int] = None,
    ) -> "Tool":
        """Build a ``Tool`` from a callable with automatic inspection.

        - **name** defaults to the callable's ``__qualname__`` (dots
          replaced with underscores).
        - **description** falls back to the callable's docstring.
        - A ``@streamable`` decorator on the callable is detected
          automatically when *streaming_config* is not provided.
        """
        qualified_name = (name or executable.__qualname__).replace(".", "_")
        resolved_description = description or inspect.getdoc(executable) or ""
        detected_streaming_config = getattr(executable, "_streaming_config", None)
        if not isinstance(detected_streaming_config, ToolStreamingConfig):
            detected_streaming_config = None

        return cls(
            name=qualified_name,
            description=resolved_description,
            executable=executable,
            streaming_config=streaming_config or detected_streaming_config,
            max_output_tokens=max_output_tokens,
        )

    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Returns the default values for all parameters that have them."""
        defaults: Dict[str, Any] = {}
        sig = inspect.signature(self.executable)
        for name, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                defaults[name] = param.default
        return defaults

    def get_parameters_spec(self) -> Dict[str, Any]:
        """Returns a JSON schema of the parameters of the tool."""
        if self.parameters_spec:
            return self.parameters_spec
        # If schema is not provided, generate it from the function signature
        # and annotations.
        schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

        signature = inspect.signature(self.executable)
        for name, param in signature.parameters.items():
            if name.startswith("_"):
                continue
            param_schema = {}
            param_type = param.annotation

            if param_type is not inspect.Parameter.empty:
                param_schema = _get_json_type(param_type)
            # Add to schema properties
            schema["properties"][name] = param_schema

            # Add to required list if no default value
            if param.default == inspect.Parameter.empty and not (
                is_union(get_origin(param_type)) and type(None) in get_args(param_type)
            ):
                schema["required"].append(name)

        return schema


def _parse_params(signature: inspect.Signature, params: Optional[Dict[str, Any]]):
    exec_params = dict(params) if params else {}

    for param_name, param in signature.parameters.items():
        if param_name in exec_params:
            param_annotation = param.annotation
            origin_type = get_origin(param_annotation)

            # Handle Union (Optional) types
            if is_union(origin_type):
                # Get the arguments of the Union type (e.g., for Optional[List[Model]]
                # -> [List[Model], None])
                union_args = get_args(param_annotation)
                # Find non-None types in the union
                non_none_types = [t for t in union_args if t is not type(None)]

                # If there's only one non-None type, it's an Optional[...] pattern
                if len(non_none_types) == 1:
                    inner_type = non_none_types[0]
                    inner_origin = get_origin(inner_type)

                    # Handle Optional[List[...]]
                    if inner_origin is list or inner_origin is List:
                        list_item_type = (
                            get_args(inner_type)[0] if get_args(inner_type) else None
                        )
                        if list_item_type and hasattr(list_item_type, "__fields__"):
                            # Convert each item in the list if it exists
                            # (Optional might be None)
                            if isinstance(exec_params[param_name], list):
                                exec_params[param_name] = [
                                    (
                                        list_item_type(**item)
                                        if isinstance(item, dict)
                                        else item
                                    )
                                    for item in exec_params[param_name]
                                ]
                    # Handle Optional[BaseModel]
                    elif hasattr(inner_type, "__fields__"):
                        if isinstance(exec_params[param_name], dict):
                            exec_params[param_name] = inner_type(
                                **exec_params[param_name]
                            )

            # Original handling for non-Optional types
            elif origin_type is list or origin_type is List:
                # Check if the list items are Pydantic models
                list_item_type = (
                    get_args(param_annotation)[0]
                    if get_args(param_annotation)
                    else None
                )
                if list_item_type and hasattr(list_item_type, "__fields__"):
                    # Convert each item in the list
                    if isinstance(exec_params[param_name], list):
                        exec_params[param_name] = [
                            list_item_type(**item) if isinstance(item, dict) else item
                            for item in exec_params[param_name]
                        ]
            elif hasattr(param_annotation, "__fields__"):
                # Handle single Pydantic model
                if isinstance(exec_params[param_name], dict):
                    exec_params[param_name] = param_annotation(
                        **exec_params[param_name]
                    )

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
        streaming_config: Optional[ToolStreamingConfig] = None,
        max_output_tokens: Optional[int] = None,
    ):
        tool = Tool.from_callable(
            executable=executable,
            name=name,
            description=description,
            streaming_config=streaming_config,
            max_output_tokens=max_output_tokens,
        )
        self.tools_index[tool.name] = tool

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
