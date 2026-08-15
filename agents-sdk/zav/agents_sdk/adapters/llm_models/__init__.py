from zav.agents_sdk.adapters.llm_models.context_window_manager import (
    ContextWindowConfiguration,
    ContextWindowManager,
    ContextWindowManagerFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(ContextWindowManagerFactory)

__all__ = [
    "AgentDependencyRegistry",
    "ContextWindowConfiguration",
    "ContextWindowManager",
    "ContextWindowManagerFactory",
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
        ResumableZAVChatCompletionClient,
        ResumableZAVChatCompletionClientFactory,
        ToolCallRequest,
        ToolCallResponse,
        ZAVChatCompletionClient,
        ZAVChatCompletionClientFactory,
    )

    AgentDependencyRegistry.register(ZAVChatCompletionClientFactory)
    AgentDependencyRegistry.register(ResumableZAVChatCompletionClientFactory)
    __all__ += [
        "ResumableZAVChatCompletionClient",
        "ResumableZAVChatCompletionClientFactory",
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

try:
    from zav.agents_sdk.adapters.llm_models.replay_chat_completion_client import (
        ReplayResumableZAVChatCompletionClient,
        ReplayResumableZAVChatCompletionClientFactory,
    )

    AgentDependencyRegistry.register(ReplayResumableZAVChatCompletionClientFactory)
    __all__ += [
        "ReplayResumableZAVChatCompletionClient",
        "ReplayResumableZAVChatCompletionClientFactory",
    ]
except ImportError:
    pass

try:
    from zav.agents_sdk.adapters.llm_models.ragelo_llm_provider import (
        ZAVRageloLLMProvider,
    )

    __all__ += ["ZAVRageloLLMProvider"]
except ImportError:
    pass
