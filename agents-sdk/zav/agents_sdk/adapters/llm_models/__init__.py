import importlib.util

from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

__all__ = [
    "AgentDependencyRegistry",
]


if importlib.util.find_spec("langchain_anthropic") is not None:
    from zav.agents_sdk.adapters.llm_models.langchain_chat_anthropic import (
        ChatAnthropicFactory,
    )

    AgentDependencyRegistry.register(ChatAnthropicFactory)
    __all__ += ["ChatAnthropicFactory"]

if importlib.util.find_spec("langchain_aws") is not None:
    from zav.agents_sdk.adapters.llm_models.langchain_chat_bedrock import (
        ChatBedrockFactory,
    )

    AgentDependencyRegistry.register(ChatBedrockFactory)
    __all__ += ["ChatBedrockFactory"]
if importlib.util.find_spec("langchain_openai") is not None:
    from zav.agents_sdk.adapters.llm_models.langchain_chat_openai import (
        ChatOpenAIFactory,
    )

    AgentDependencyRegistry.register(ChatOpenAIFactory)
    __all__ += ["ChatOpenAIFactory"]
if importlib.util.find_spec("zav.prompt_completion") is not None:
    from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import (
        ChatCompletion,
        ChatCompletionSender,
        ChatResponse,
        FunctionCallResponse,
        ToolCallRequest,
        ToolCallResponse,
        ZAVChatCompletionClient,
        ZAVChatCompletionClientFactory,
    )

    AgentDependencyRegistry.register(ZAVChatCompletionClientFactory)
    __all__ += [
        "ZAVChatCompletionClient",
        "ZAVChatCompletionClientFactory",
        "ChatCompletion",
        "ChatCompletionSender",
        "ChatResponse",
        "FunctionCallResponse",
        "ToolCallRequest",
        "ToolCallResponse",
    ]
