from zav.agents_sdk.adapters.context.context_provider import (
    ContextConfiguration,
    ContextProvider,
    ContextProviderFactory,
)
from zav.agents_sdk.adapters.context.context_source import (
    ContextOrigin,
    ContextSource,
    ContextSourceGroup,
    ResolvedContextItem,
)
from zav.agents_sdk.adapters.context.context_sources import (
    CustomContextSource,
    CustomContextSourceFactory,
    DocumentContextSource,
    DocumentContextSourceFactory,
    TagContextSource,
    TagContextSourceFactory,
)
from zav.agents_sdk.adapters.context.filter_context_source import (
    FilterContextSource,
    FilterContextSourceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(ContextProviderFactory)
AgentDependencyRegistry.register(DocumentContextSourceFactory)
AgentDependencyRegistry.register(TagContextSourceFactory)
AgentDependencyRegistry.register(CustomContextSourceFactory)
AgentDependencyRegistry.register(FilterContextSourceFactory)

__all__ = [
    "ContextConfiguration",
    "ContextOrigin",
    "ContextProvider",
    "ContextProviderFactory",
    "ContextSource",
    "ContextSourceGroup",
    "CustomContextSource",
    "CustomContextSourceFactory",
    "DocumentContextSource",
    "DocumentContextSourceFactory",
    "FilterContextSource",
    "FilterContextSourceFactory",
    "ResolvedContextItem",
    "TagContextSource",
    "TagContextSourceFactory",
]
