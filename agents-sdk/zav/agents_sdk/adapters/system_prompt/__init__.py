from zav.agents_sdk.adapters.system_prompt.sources.disk_system_prompt_source import (
    DiskSystemPromptSource,
    DiskSystemPromptSourceConfiguration,
    DiskSystemPromptSourceFactory,
)
from zav.agents_sdk.adapters.system_prompt.sources.in_memory_system_prompt_source import (  # noqa: E501
    InMemorySystemPromptSource,
    InMemorySystemPromptSourceConfiguration,
    InMemorySystemPromptSourceFactory,
)
from zav.agents_sdk.adapters.system_prompt.system_prompt_provider import (
    SystemPromptProvider,
    SystemPromptProviderConfiguration,
    SystemPromptProviderFactory,
)
from zav.agents_sdk.adapters.system_prompt.system_prompt_source import (
    SystemPromptSource,
    SystemPromptSourceGroup,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(SystemPromptProviderFactory)
AgentDependencyRegistry.register(DiskSystemPromptSourceFactory)
AgentDependencyRegistry.register(InMemorySystemPromptSourceFactory)

__all__ = [
    "DiskSystemPromptSource",
    "DiskSystemPromptSourceConfiguration",
    "DiskSystemPromptSourceFactory",
    "InMemorySystemPromptSource",
    "InMemorySystemPromptSourceConfiguration",
    "InMemorySystemPromptSourceFactory",
    "SystemPromptProvider",
    "SystemPromptProviderConfiguration",
    "SystemPromptProviderFactory",
    "SystemPromptSource",
    "SystemPromptSourceGroup",
]
