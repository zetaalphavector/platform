from typing import Callable, Optional

from zav.llm_tracing import Span
from zav.prompt_completion import (
    ImageGenerationClient,
    ImageGenerationRequest,
    ImageGenerationResult,
)

from zav.agents_sdk.adapters.policies.llm_selection import LLMSelectionConfiguration
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.llm_client_factory import LLMClientFactory, LLMNotConfigured


class _DeferredImageGenerationClient(ImageGenerationClient):
    """Resolves the LLM selection on first use, so building the client never
    requires a resolvable selection (tool listing must succeed with no LLM in
    scope)."""

    def __init__(self, builder: Callable[[], ImageGenerationClient]):
        self.__builder = builder
        self.__client: Optional[ImageGenerationClient] = None

    @classmethod
    def from_configuration(
        cls,
        vendor_configuration,
        model_configuration,
        span: Optional[Span] = None,
    ) -> ImageGenerationClient:
        raise NotImplementedError

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        if self.__client is None:
            self.__client = self.__builder()
        return await self.__client.generate(request)


def _deferred_image_generation_client(
    llm_client_factory: Optional[LLMClientFactory],
    llm_selection_configuration: LLMSelectionConfiguration,
    span: Optional[Span],
) -> ImageGenerationClient:
    def build() -> ImageGenerationClient:
        if llm_client_factory is None:
            raise LLMNotConfigured("No LLM client factory in scope.")
        return llm_client_factory.create_image_generation_client(
            name=llm_selection_configuration.llm_configuration_name, span=span
        )

    return _DeferredImageGenerationClient(build)


class ImageGenerationClientFactory(AgentDependencyFactory):
    """`image_llm_selection_configuration` names the LLM configuration (type
    `image`) to generate with. It is an admin setting resolved by name, outside
    the chat allow-list."""

    @classmethod
    def create(
        cls,
        llm_client_factory: Optional[LLMClientFactory] = None,
        image_llm_selection_configuration: LLMSelectionConfiguration = (
            LLMSelectionConfiguration()
        ),
        span: Optional[Span] = None,
    ) -> ImageGenerationClient:
        return _deferred_image_generation_client(
            llm_client_factory, image_llm_selection_configuration, span
        )


__all__ = ["ImageGenerationClientFactory"]
