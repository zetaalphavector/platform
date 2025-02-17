import inspect
import json
from typing import Any, Generic, TypeVar, cast

try:
    from pydantic.v1.json import pydantic_encoder
except ImportError:
    from pydantic.json import pydantic_encoder  # type: ignore

from zav.llm_tracing.trace import Span

T_ = TypeVar("T_")


class Instrumented(Generic[T_]):
    def __init__(self, instance, span: Span):
        self.instance = instance
        self.span = span

    def __getattr__(self, name: str) -> Any:
        original_attr = getattr(self.instance, name)

        if name.startswith("__") or name.startswith("_"):
            return original_attr

        if callable(original_attr):
            qualified_name = original_attr.__qualname__.replace(".", "_")

            def new_func(*args, **kwargs):
                sig = inspect.signature(original_attr)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                # Code to run before the actual method call
                observation = self.span.new(
                    name=qualified_name,
                    attributes={
                        "input": json.loads(
                            json.dumps(
                                dict(bound_args.arguments), default=pydantic_encoder
                            )
                        )
                    },
                )

                # Call the original method
                result = original_attr(*args, **kwargs)

                # Code to run after the actual method call
                observation.end(attributes={"output": result})
                return result

            return new_func
        else:
            return original_attr


def instrument_instance(instance: T_, span: Span) -> T_:
    return cast(T_, Instrumented(instance, span))
