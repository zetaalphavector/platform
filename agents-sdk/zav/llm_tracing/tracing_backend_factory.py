from typing import Callable, Dict, Type

from zav.llm_tracing.trace import TracingBackend
from zav.llm_tracing.tracing_configuration import TracingConfiguration


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
    def create(cls, config: TracingConfiguration) -> TracingBackend:
        vendor_name = config.vendor.value
        vendor_configuration = getattr(config.vendor_configuration, vendor_name, None)
        if not vendor_configuration:
            raise ValueError(f"Vendor configuration not found for: {config.vendor}")
        if vendor_name not in cls.registry:
            raise ValueError(f"Unknown tracing vendor: {vendor_name}")
        return cls.registry[vendor_name](vendor_configuration)
