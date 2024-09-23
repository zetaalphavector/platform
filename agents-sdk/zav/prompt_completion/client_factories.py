from zav.prompt_completion.base_factory import BaseClientFactory
from zav.prompt_completion.client import (
    ChatCompletionClient,
    PromptCompletionClient,
    PromptCompletionWithLogitsClient,
)


class ChatClientFactory(BaseClientFactory[ChatCompletionClient]): ...


class PromptClientFactory(BaseClientFactory[PromptCompletionClient]): ...


class PromptWithLogitsClientFactory(
    BaseClientFactory[PromptCompletionWithLogitsClient]
): ...
