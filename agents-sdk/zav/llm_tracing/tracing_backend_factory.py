from typing import Callable, Dict, Type

from zav.llm_tracing.trace import TracingBackend


class TracingBackendFactory:
    registry: Dict[str, Type[TracingBackend]] = {}

    @classmethod
    def register(cls, vendor_name: str) -> Callable:
        def inner_wrapper(
            tracing_backend: Type[TracingBackend],
        ) -> Type[TracingBackend]:
            cls.registry[vendor_name] = tracing_backend
            return tracing_backend

        return inner_wrapper

    @classmethod
    def create(cls, vendor_name: str, config) -> TracingBackend:
        if vendor_name not in cls.registry:
            raise ValueError(f"Unknown tracing vendor: {vendor_name}")
        return cls.registry[vendor_name](**config)
