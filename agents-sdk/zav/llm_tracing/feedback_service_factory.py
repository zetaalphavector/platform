from typing import Callable, Dict, Type

from zav.llm_tracing.feedback import FeedbackService
from zav.llm_tracing.tracing_configuration import TracingConfiguration


class FeedbackServiceFactory:
    registry: Dict[str, Type[FeedbackService]] = {}

    @classmethod
    def register(cls, vendor_name: str) -> Callable:
        def inner_wrapper(
            feedback_service: Type[FeedbackService],
        ) -> Type[FeedbackService]:
            cls.registry[vendor_name] = feedback_service
            return feedback_service

        return inner_wrapper

    @classmethod
    def create(cls, config: TracingConfiguration) -> FeedbackService:
        vendor_name = config.vendor.value
        vendor_configuration = getattr(config.vendor_configuration, vendor_name, None)
        if not vendor_configuration:
            raise ValueError(f"Vendor configuration not found for: {config.vendor}")
        if vendor_name not in cls.registry:
            raise ValueError(f"Unknown feedback service vendor: {vendor_name}")
        return cls.registry[vendor_name](vendor_configuration)
