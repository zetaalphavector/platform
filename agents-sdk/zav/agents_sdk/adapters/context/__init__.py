from zav.agents_sdk.adapters.context.context_provider import (
    ContextProvider,
    ContextProviderConfiguration,
    ContextProviderFactory,
)
from zav.agents_sdk.adapters.context.context_source import (
    ContextOrigin,
    ContextSource,
    ContextSourceGroup,
    ResolvedContextItem,
)
from zav.agents_sdk.adapters.context.sources.custom_context_source import (
    CustomContextSource,
    CustomContextSourceConfiguration,
    CustomContextSourceFactory,
)
from zav.agents_sdk.adapters.context.sources.document_context_source import (
    DocumentContextSource,
    DocumentContextSourceConfiguration,
    DocumentContextSourceFactory,
)
from zav.agents_sdk.adapters.context.sources.filter_context_source import (
    FilterContextSource,
    FilterContextSourceConfiguration,
    FilterContextSourceFactory,
)
from zav.agents_sdk.adapters.context.sources.image_context_source import (
    ImageContextSource,
    ImageContextSourceConfiguration,
    ImageContextSourceFactory,
)
from zav.agents_sdk.adapters.context.sources.scheduled_task_context_source import (
    ScheduledTaskContextSource,
    ScheduledTaskContextSourceConfiguration,
    ScheduledTaskContextSourceFactory,
)
from zav.agents_sdk.adapters.context.sources.tag_context_source import (
    TagContextSource,
    TagContextSourceConfiguration,
    TagContextSourceFactory,
)
from zav.agents_sdk.adapters.context.sources.user_document_context_source import (
    UserDocumentContextSource,
    UserDocumentContextSourceConfiguration,
    UserDocumentContextSourceFactory,
)
from zav.agents_sdk.adapters.context.sources.user_workspace_context_source import (
    UserWorkspaceContextSource,
    UserWorkspaceContextSourceConfiguration,
    UserWorkspaceContextSourceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(ContextProviderFactory)
AgentDependencyRegistry.register(DocumentContextSourceFactory)
AgentDependencyRegistry.register(TagContextSourceFactory)
AgentDependencyRegistry.register(CustomContextSourceFactory)
AgentDependencyRegistry.register(FilterContextSourceFactory)
AgentDependencyRegistry.register(ScheduledTaskContextSourceFactory)
AgentDependencyRegistry.register(ImageContextSourceFactory)
AgentDependencyRegistry.register(UserDocumentContextSourceFactory)
AgentDependencyRegistry.register(UserWorkspaceContextSourceFactory)

__all__ = [
    "ContextProviderConfiguration",
    "ContextOrigin",
    "ContextProvider",
    "ContextProviderFactory",
    "ContextSource",
    "ContextSourceGroup",
    "CustomContextSource",
    "CustomContextSourceConfiguration",
    "CustomContextSourceFactory",
    "DocumentContextSource",
    "DocumentContextSourceConfiguration",
    "DocumentContextSourceFactory",
    "FilterContextSource",
    "FilterContextSourceConfiguration",
    "FilterContextSourceFactory",
    "ImageContextSource",
    "ImageContextSourceConfiguration",
    "ImageContextSourceFactory",
    "ResolvedContextItem",
    "ScheduledTaskContextSource",
    "ScheduledTaskContextSourceConfiguration",
    "ScheduledTaskContextSourceFactory",
    "TagContextSource",
    "TagContextSourceConfiguration",
    "TagContextSourceFactory",
    "UserDocumentContextSource",
    "UserDocumentContextSourceConfiguration",
    "UserDocumentContextSourceFactory",
    "UserWorkspaceContextSource",
    "UserWorkspaceContextSourceConfiguration",
    "UserWorkspaceContextSourceFactory",
]
