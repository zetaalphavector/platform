from zav.agents_sdk.adapters.instructions.instruction_source import (
    InstructionSource,
    InstructionSourceGroup,
)
from zav.agents_sdk.adapters.instructions.instructions_provider import (
    InstructionsProvider,
    InstructionsProviderConfiguration,
    InstructionsProviderFactory,
)
from zav.agents_sdk.adapters.instructions.sources.citation_instruction_source import (
    CitableResource,
    CitationInstructionSource,
    CitationInstructionSourceConfiguration,
    CitationInstructionSourceFactory,
    CitationStrategy,
    build_citation_instructions,
)
from zav.agents_sdk.adapters.instructions.sources.datetime_instruction_source import (
    DateTimeInstructionSource,
    DateTimeInstructionSourceConfiguration,
    DateTimeInstructionSourceFactory,
)
from zav.agents_sdk.adapters.instructions.sources.in_memory_instruction_source import (
    InMemoryInstructionDefinition,
    InMemoryInstructionSource,
    InMemoryInstructionSourceConfiguration,
    InMemoryInstructionSourceFactory,
)
from zav.agents_sdk.adapters.instructions.sources.index_explorer_instruction_source import (
    IndexExplorerInstructionSource,
    IndexExplorerInstructionSourceConfiguration,
    IndexExplorerInstructionSourceFactory,
)
from zav.agents_sdk.adapters.instructions.sources.onboarding_instruction_source import (
    OnboardingInstructionSource,
    OnboardingInstructionSourceConfiguration,
    OnboardingInstructionSourceFactory,
)
from zav.agents_sdk.adapters.instructions.sources.platform_docs_instruction_source import (
    PlatformDocsInstructionSource,
    PlatformDocsInstructionSourceConfiguration,
    PlatformDocsInstructionSourceFactory,
)
from zav.agents_sdk.adapters.instructions.sources.user_instruction_source import (
    UserInstructionSource,
    UserInstructionSourceConfiguration,
    UserInstructionSourceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(InstructionsProviderFactory)
AgentDependencyRegistry.register(CitationInstructionSourceFactory)
AgentDependencyRegistry.register(DateTimeInstructionSourceFactory)
AgentDependencyRegistry.register(IndexExplorerInstructionSourceFactory)
AgentDependencyRegistry.register(InMemoryInstructionSourceFactory)
AgentDependencyRegistry.register(OnboardingInstructionSourceFactory)
AgentDependencyRegistry.register(PlatformDocsInstructionSourceFactory)
AgentDependencyRegistry.register(UserInstructionSourceFactory)

__all__ = [
    "CitableResource",
    "CitationInstructionSource",
    "CitationInstructionSourceConfiguration",
    "CitationInstructionSourceFactory",
    "CitationStrategy",
    "DateTimeInstructionSource",
    "DateTimeInstructionSourceConfiguration",
    "DateTimeInstructionSourceFactory",
    "IndexExplorerInstructionSource",
    "IndexExplorerInstructionSourceConfiguration",
    "IndexExplorerInstructionSourceFactory",
    "InMemoryInstructionDefinition",
    "InMemoryInstructionSource",
    "InMemoryInstructionSourceConfiguration",
    "InMemoryInstructionSourceFactory",
    "InstructionSource",
    "InstructionSourceGroup",
    "InstructionsProvider",
    "InstructionsProviderConfiguration",
    "InstructionsProviderFactory",
    "OnboardingInstructionSource",
    "OnboardingInstructionSourceConfiguration",
    "OnboardingInstructionSourceFactory",
    "PlatformDocsInstructionSource",
    "PlatformDocsInstructionSourceConfiguration",
    "PlatformDocsInstructionSourceFactory",
    "UserInstructionSource",
    "UserInstructionSourceConfiguration",
    "UserInstructionSourceFactory",
    "build_citation_instructions",
]
