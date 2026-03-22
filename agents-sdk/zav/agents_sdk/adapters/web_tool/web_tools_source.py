from typing import ClassVar, List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.adapters.web_tool.url_crawler import WebPageCrawler
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool

_WEB_FETCH_DESCRIPTION = """\
Fetch a web page and return its content as text. Use this tool when you need \
live information from a specific URL — documentation, articles, API responses, \
or any publicly accessible web page.

The tool fetches the page, strips boilerplate (scripts, styles, navigation), \
and returns clean readable content. Optionally provide a prompt to extract \
only the relevant parts.\
"""

_SYSTEM_PROMPT_SECTION = """\
## Web fetching

You have access to a `web_fetch` tool that retrieves live content from URLs.

- Use it when you need current information from a specific web page.
- Do not guess or hallucinate URLs — only fetch URLs the user provided, \
URLs you found in documents, or well-known canonical URLs.
- When the user asks about the content of a link, fetch it rather than \
speculating.
- If a page fails to load, inform the user instead of fabricating content.\
"""


class WebToolsConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable web tools.")
    output_markdown: bool = Field(
        True, description="Return content as markdown instead of plain text."
    )
    max_content_chars: Optional[int] = Field(
        None,
        description="Truncate fetched content to this many characters.",
    )
    timeout_seconds: Optional[float] = Field(
        30.0, description="Timeout for each fetch request."
    )
    cache_max_size: int = Field(50, description="Maximum number of cached responses.")
    cache_ttl_seconds: float = Field(900.0, description="Cache entry TTL in seconds.")
    upgrade_to_https: bool = Field(
        True, description="Upgrade http:// URLs to https://."
    )
    blocked_hostnames: Optional[List[str]] = Field(
        None,
        description="Additional hostnames to block from crawling.",
    )


class WebToolsSource(ToolsSource):
    source_name: ClassVar[str] = "web_tools"

    def __init__(
        self,
        crawler: WebPageCrawler,
        enabled: bool,
    ) -> None:
        self.__crawler = crawler
        self.__enabled = enabled

    async def get_tools(self) -> List[Tool]:
        if not self.__enabled:
            return []

        crawler = self.__crawler

        async def web_fetch(url: str, prompt: Optional[str] = None) -> str:
            """Fetch a web page and return its content.

            Args:
                url: The URL to fetch.
                prompt: Optional extraction prompt — when provided, an LLM
                    filters the page content to only the parts relevant to
                    this prompt.
            """
            result = await crawler.safe_crawl(url, prompt=prompt)
            if result is None:
                return f"Failed to fetch content from {url}."
            if not result.strip():
                return f"The page at {url} returned no readable content."
            return result

        return [
            Tool.from_callable(
                name="web_fetch",
                executable=web_fetch,
                description=_WEB_FETCH_DESCRIPTION,
            ),
        ]

    def to_prompt(self) -> str:
        if not self.__enabled:
            return ""
        return _SYSTEM_PROMPT_SECTION


class WebToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        web_tools_configuration: WebToolsConfiguration = WebToolsConfiguration(),
    ) -> WebToolsSource:
        crawler = WebPageCrawler(
            output_markdown=web_tools_configuration.output_markdown,
            max_content_chars=web_tools_configuration.max_content_chars,
            timeout_seconds=web_tools_configuration.timeout_seconds,
            cache_max_size=web_tools_configuration.cache_max_size,
            cache_ttl_seconds=web_tools_configuration.cache_ttl_seconds,
            upgrade_to_https=web_tools_configuration.upgrade_to_https,
            blocked_hostnames=web_tools_configuration.blocked_hostnames,
        )
        return WebToolsSource(
            crawler=crawler,
            enabled=web_tools_configuration.enabled,
        )
