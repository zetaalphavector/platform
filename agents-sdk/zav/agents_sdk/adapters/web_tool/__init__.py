from zav.agents_sdk.adapters.web_tool.url_crawler import (
    LLMExtractor,
    WebPageCrawler,
    parse_html,
)
from zav.agents_sdk.adapters.web_tool.web_tools_source import (
    WebToolsConfiguration,
    WebToolsSource,
    WebToolsSourceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(WebToolsSourceFactory)

__all__ = [
    "LLMExtractor",
    "WebPageCrawler",
    "WebToolsConfiguration",
    "WebToolsSource",
    "WebToolsSourceFactory",
    "parse_html",
]
