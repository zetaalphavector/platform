from zav.agents_sdk.adapters.tools.sources.dataframe_tools_source import (
    DataFrameToolsSource,
    DataFrameToolsSourceConfiguration,
    DataFrameToolsSourceFactory,
)
from zav.agents_sdk.adapters.tools.sources.document_tools_source import (
    DocumentTools,
    DocumentToolsFactory,
    DocumentToolsSourceConfiguration,
)
from zav.agents_sdk.adapters.tools.sources.index_tools_source import (
    IndexToolsSource,
    IndexToolsSourceConfiguration,
    IndexToolsSourceFactory,
    MetadataFieldConfig,
    extract_hit_metadata,
)
from zav.agents_sdk.adapters.tools.sources.note_tools_source import (
    NoteToolsFactory,
    NoteToolsSource,
    NoteToolsSourceConfiguration,
)
from zav.agents_sdk.adapters.tools.sources.onboarding_tools_source import (
    OnboardingStep,
    OnboardingToolsSource,
    OnboardingToolsSourceConfiguration,
    OnboardingToolsSourceFactory,
)
from zav.agents_sdk.adapters.tools.sources.platform_docs_tools_source import (
    PlatformDocsToolsSource,
    PlatformDocsToolsSourceConfiguration,
    PlatformDocsToolsSourceFactory,
)
from zav.agents_sdk.adapters.tools.sources.plot_tools_source import (
    PlotSeries,
    PlotToolsSource,
    PlotToolsSourceConfiguration,
    PlotToolsSourceFactory,
)
from zav.agents_sdk.adapters.tools.sources.sub_agent_tool_source import (
    SubAgentToolSource,
    SubAgentToolSourceConfiguration,
    SubAgentToolSourceFactory,
)
from zav.agents_sdk.adapters.tools.sources.tag_tools_source import (
    TagTools,
    TagToolsFactory,
    TagToolsSourceConfiguration,
)
from zav.agents_sdk.adapters.tools.sources.url_crawler import (
    LLMExtractor,
    WebPageCrawler,
    parse_html,
)
from zav.agents_sdk.adapters.tools.sources.user_document_tools_source import (
    UserDocumentToolsSource,
    UserDocumentToolsSourceConfiguration,
    UserDocumentToolsSourceFactory,
)
from zav.agents_sdk.adapters.tools.sources.web_tools_source import (
    WebToolsSource,
    WebToolsSourceConfiguration,
    WebToolsSourceFactory,
)
from zav.agents_sdk.adapters.tools.tools_provider import (
    ToolsProvider,
    ToolsProviderConfiguration,
    ToolsProviderFactory,
)
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource, ToolsSourceGroup
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(ToolsProviderFactory)
AgentDependencyRegistry.register(DataFrameToolsSourceFactory)
AgentDependencyRegistry.register(DocumentToolsFactory)
AgentDependencyRegistry.register(IndexToolsSourceFactory)
AgentDependencyRegistry.register(NoteToolsFactory)
AgentDependencyRegistry.register(TagToolsFactory)
AgentDependencyRegistry.register(UserDocumentToolsSourceFactory)
AgentDependencyRegistry.register(WebToolsSourceFactory)
AgentDependencyRegistry.register(OnboardingToolsSourceFactory)
AgentDependencyRegistry.register(PlatformDocsToolsSourceFactory)
AgentDependencyRegistry.register(PlotToolsSourceFactory)
AgentDependencyRegistry.register(SubAgentToolSourceFactory)

try:
    from zav.agents_sdk.adapters.tools.sources.sql_database_tools_source import (
        AggregateOperation,
        AggregateParams,
        FilterParams,
        JoinParams,
        JoinType,
        OperationType,
        QueryPlan,
        SelectColumnsParams,
        SortOrder,
        SortParams,
        SQLDatabaseToolsSource,
        SQLDatabaseToolsSourceConfiguration,
        SQLDatabaseToolsSourceFactory,
        SQLFilterCondition,
        SQLFilterOperation,
    )

    AgentDependencyRegistry.register(SQLDatabaseToolsSourceFactory)
except ImportError:
    pass

try:
    from zav.agents_sdk.adapters.tools.sources.image_search_tools_source import (
        ImageToolsSource,
        ImageToolsSourceConfiguration,
        ImageToolsSourceFactory,
    )

    AgentDependencyRegistry.register(ImageToolsSourceFactory)
except ImportError:
    pass

__all__ = [
    "DocumentTools",
    "DocumentToolsSourceConfiguration",
    "DocumentToolsFactory",
    "IndexToolsSource",
    "IndexToolsSourceConfiguration",
    "IndexToolsSourceFactory",
    "MetadataFieldConfig",
    "NoteToolsSource",
    "NoteToolsSourceConfiguration",
    "NoteToolsFactory",
    "LLMExtractor",
    "OnboardingStep",
    "OnboardingToolsSource",
    "OnboardingToolsSourceConfiguration",
    "OnboardingToolsSourceFactory",
    "PlotSeries",
    "PlotToolsSource",
    "PlotToolsSourceConfiguration",
    "PlotToolsSourceFactory",
    "SubAgentToolSource",
    "SubAgentToolSourceConfiguration",
    "SubAgentToolSourceFactory",
    "TagTools",
    "TagToolsSourceConfiguration",
    "TagToolsFactory",
    "ToolsProviderConfiguration",
    "ToolsProvider",
    "ToolsProviderFactory",
    "ToolsSource",
    "ToolsSourceGroup",
    "UserDocumentToolsSource",
    "UserDocumentToolsSourceConfiguration",
    "UserDocumentToolsSourceFactory",
    "WebPageCrawler",
    "WebToolsSource",
    "WebToolsSourceConfiguration",
    "WebToolsSourceFactory",
    "extract_hit_metadata",
    "parse_html",
]
