from typing import Callable, Dict, Generic, Optional, Tuple, Type, TypeVar

from zav.llm_domain import LLMClientConfiguration, LLMModelType, LLMProviderName
from zav.llm_tracing import Span

from zav.prompt_completion.client import BaseCompletionClient

PROMPT_COMPLETION_CLIENT = TypeVar(
    "PROMPT_COMPLETION_CLIENT", bound=BaseCompletionClient
)


class BaseClientFactory(Generic[PROMPT_COMPLETION_CLIENT]):
    registry: Dict[
        Tuple[LLMProviderName, LLMModelType],
        Type[PROMPT_COMPLETION_CLIENT],
    ] = {}

    def __init_subclass__(cls):
        cls.registry = {}

    @classmethod
    def register(
        cls,
        provider_name: LLMProviderName,
        model_type: LLMModelType,
    ) -> Callable:
        def prompt_inner_wrapper(
            wrapped_class: Type[PROMPT_COMPLETION_CLIENT],
        ) -> Type[PROMPT_COMPLETION_CLIENT]:
            cls.registry[(provider_name, model_type)] = wrapped_class
            return wrapped_class

        return prompt_inner_wrapper

    @classmethod
    def create(
        cls,
        config: LLMClientConfiguration,
        span: Optional[Span] = None,
    ) -> PROMPT_COMPLETION_CLIENT:
        vendor_configuration = getattr(
            config.vendor_configuration, config.vendor.value, None
        )
        if not vendor_configuration:
            raise ValueError(f"Vendor configuration not found for: {config.vendor}")
        return cls.registry[
            (config.vendor, config.model_configuration.type)
        ].from_configuration(
            vendor_configuration=vendor_configuration,
            model_configuration=config.model_configuration,
            span=span,
        )
