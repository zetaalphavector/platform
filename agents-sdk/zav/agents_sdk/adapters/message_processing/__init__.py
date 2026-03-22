from zav.agents_sdk.adapters.message_processing.citation_processor import (
    CitationProcessor,
    CitationProcessorFactory,
)
from zav.agents_sdk.adapters.message_processing.message_processing_provider import (
    MessageProcessingProvider,
    MessageProcessingProviderFactory,
)
from zav.agents_sdk.adapters.message_processing.message_processor import (
    MessageProcessor,
    MessageProcessorGroup,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(CitationProcessorFactory)
AgentDependencyRegistry.register(MessageProcessingProviderFactory)

__all__ = [
    "CitationProcessor",
    "CitationProcessorFactory",
    "MessageProcessingProvider",
    "MessageProcessingProviderFactory",
    "MessageProcessor",
    "MessageProcessorGroup",
]
