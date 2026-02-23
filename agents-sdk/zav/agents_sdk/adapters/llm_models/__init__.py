from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

__all__ = [
    "AgentDependencyRegistry",
]


try:
    from zav.agents_sdk.adapters.llm_models.langchain_chat_anthropic import (
        ChatAnthropicFactory,
    )

    AgentDependencyRegistry.register(ChatAnthropicFactory)
    __all__ += ["ChatAnthropicFactory"]
except ImportError:
    pass

try:
    from zav.agents_sdk.adapters.llm_models.langchain_chat_bedrock import (
        ChatBedrockFactory,
    )

    AgentDependencyRegistry.register(ChatBedrockFactory)
    __all__ += ["ChatBedrockFactory"]

except ImportError:
    pass

try:
    from zav.agents_sdk.adapters.llm_models.langchain_chat_openai import (
        ChatOpenAIFactory,
    )

    AgentDependencyRegistry.register(ChatOpenAIFactory)
    __all__ += ["ChatOpenAIFactory"]
except ImportError:
    pass

try:
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
except ImportError:
    pass
