from zav.agents_sdk.adapters.message_processing.message_processing_provider import (
    MessageProcessingProvider,
    MessageProcessingProviderConfiguration,
    MessageProcessingProviderFactory,
)
from zav.agents_sdk.adapters.message_processing.message_processor import (
    MessageProcessor,
    MessageProcessorGroup,
    StreamItem,
)
from zav.agents_sdk.adapters.message_processing.processors.citation_processor import (
    CitationProcessor,
    CitationProcessorConfiguration,
    CitationProcessorFactory,
)
from zav.agents_sdk.adapters.message_processing.processors.scheduled_task_notification_processor import (
    ScheduledTaskNotificationProcessor,
    ScheduledTaskNotificationProcessorConfiguration,
    ScheduledTaskNotificationProcessorFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(CitationProcessorFactory)
AgentDependencyRegistry.register(MessageProcessingProviderFactory)
AgentDependencyRegistry.register(ScheduledTaskNotificationProcessorFactory)

__all__ = [
    "CitationProcessorConfiguration",
    "CitationProcessor",
    "CitationProcessorConfiguration",
    "CitationProcessorFactory",
    "MessageProcessingProviderConfiguration",
    "MessageProcessingProvider",
    "MessageProcessingProviderFactory",
    "MessageProcessor",
    "MessageProcessorGroup",
    "ScheduledTaskNotificationProcessorConfiguration",
    "ScheduledTaskNotificationProcessor",
    "ScheduledTaskNotificationProcessorFactory",
    "StreamItem",
]
