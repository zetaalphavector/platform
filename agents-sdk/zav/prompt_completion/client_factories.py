from typing import Optional

from zav.llm_domain import LLMClientConfiguration
from zav.llm_tracing import Span

from zav.prompt_completion.base_factory import BaseClientFactory
from zav.prompt_completion.client import (
    ChatCompletionClient,
    ImageGenerationClient,
    PromptCompletionClient,
    PromptCompletionWithLogitsClient,
)
from zav.prompt_completion.tool_search import ToolSearchChatClient


class ChatClientFactory(BaseClientFactory[ChatCompletionClient]):
    @classmethod
    def create(
        cls,
        config: LLMClientConfiguration,
        span: Optional[Span] = None,
    ) -> ChatCompletionClient:
        return ToolSearchChatClient(
            super().create(config, span=span),
            config.supported_tool_search_methods(),
            config.model_configuration.name,
        )


class PromptClientFactory(BaseClientFactory[PromptCompletionClient]): ...


class PromptWithLogitsClientFactory(
    BaseClientFactory[PromptCompletionWithLogitsClient]
): ...


class ImageClientFactory(BaseClientFactory[ImageGenerationClient]): ...
