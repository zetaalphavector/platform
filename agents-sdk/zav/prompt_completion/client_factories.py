from zav.prompt_completion.base_factory import BaseClientFactory
from zav.prompt_completion.client import (
    ChatCompletionClient,
    PromptCompletionClient,
    PromptCompletionWithLogitsClient,
)


class ChatClientFactory(BaseClientFactory[ChatCompletionClient]):
    pass


class PromptClientFactory(BaseClientFactory[PromptCompletionClient]):
    pass


class PromptWithLogitsClientFactory(
    BaseClientFactory[PromptCompletionWithLogitsClient]
):
    pass
