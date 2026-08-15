from typing import Callable, Dict, Type

from zav.llm_tracing.trace_cleanup import TraceCleanupService
from zav.llm_tracing.tracing_configuration import TracingConfiguration


class TraceCleanupServiceFactory:
    registry: Dict[str, Type[TraceCleanupService]] = {}

    @classmethod
    def register(cls, vendor_name: str) -> Callable:
        def inner_wrapper(
            cleanup_service: Type[TraceCleanupService],
        ) -> Type[TraceCleanupService]:
            cls.registry[vendor_name] = cleanup_service
            return cleanup_service

        return inner_wrapper

    @classmethod
    def create(cls, config: TracingConfiguration) -> TraceCleanupService:
        vendor_name = config.vendor.value
        vendor_configuration = getattr(config.vendor_configuration, vendor_name, None)
        if not vendor_configuration:
            raise ValueError(f"Vendor configuration not found for: {config.vendor}")
        if vendor_name not in cls.registry:
            raise ValueError(f"Unknown trace cleanup service vendor: {vendor_name}")
        return cls.registry[vendor_name](vendor_configuration)
