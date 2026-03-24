from zav.agents_sdk.adapters.search.citation import (
    CitationConfig,
    CitationStore,
    CitationStoreFactory,
    RegisteredHit,
)
from zav.agents_sdk.adapters.search.filter_translator import FilterTranslator
from zav.agents_sdk.adapters.search.index_tools_source import (
    IndexToolsConfig,
    IndexToolsSource,
    IndexToolsSourceFactory,
)
from zav.agents_sdk.adapters.search.search_skills_source import (
    SearchSkillsSource,
    SearchSkillsSourceFactory,
)
from zav.agents_sdk.adapters.search.user_document_tools_source import (
    UserDocumentToolsConfig,
    UserDocumentToolsSource,
    UserDocumentToolsSourceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(CitationStoreFactory)
AgentDependencyRegistry.register(IndexToolsSourceFactory)
AgentDependencyRegistry.register(SearchSkillsSourceFactory)
AgentDependencyRegistry.register(UserDocumentToolsSourceFactory)

__all__ = [
    "CitationConfig",
    "CitationStore",
    "CitationStoreFactory",
    "FilterTranslator",
    "RegisteredHit",
    "SearchSkillsSource",
    "SearchSkillsSourceFactory",
    "IndexToolsConfig",
    "IndexToolsSource",
    "IndexToolsSourceFactory",
    "UserDocumentToolsConfig",
    "UserDocumentToolsSource",
    "UserDocumentToolsSourceFactory",
]
