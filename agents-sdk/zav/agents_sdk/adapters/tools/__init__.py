from zav.agents_sdk.adapters.tools.document_tools import (
    DocumentTools,
    DocumentToolsConfig,
    DocumentToolsFactory,
    extract_hit_metadata,
)
from zav.agents_sdk.adapters.tools.tag_tools import (
    TagTools,
    TagToolsConfig,
    TagToolsFactory,
)
from zav.agents_sdk.adapters.tools.tools_provider import (
    ToolsConfiguration,
    ToolsProvider,
    ToolsProviderFactory,
)
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource, ToolsSourceGroup
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(ToolsProviderFactory)
AgentDependencyRegistry.register(DocumentToolsFactory)
AgentDependencyRegistry.register(TagToolsFactory)

__all__ = [
    "DocumentTools",
    "DocumentToolsConfig",
    "DocumentToolsFactory",
    "TagTools",
    "TagToolsConfig",
    "TagToolsFactory",
    "ToolsConfiguration",
    "ToolsProvider",
    "ToolsProviderFactory",
    "ToolsSource",
    "ToolsSourceGroup",
    "extract_hit_metadata",
]
