from typing import Any, Dict, List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.tools.sources.index_tools_source import IndexToolsSource
from zav.agents_sdk.adapters.tools.sources.url_crawler import WebPageCrawler
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable

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


class WebToolsSourceConfiguration(BaseModel):
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
    source_name = "web_tools"

    def __init__(
        self,
        crawler: WebPageCrawler,
        enabled: bool,
        index_tools: IndexToolsSource,
    ) -> None:
        self.__crawler = crawler
        self.__index_tools = index_tools
        self.enabled = enabled
        self.__federated_indexes: Optional[List[Dict[str, Any]]] = None

    async def __get_federated_indexes(self) -> List[Dict[str, Any]]:
        if self.__federated_indexes is None:
            self.__federated_indexes = await self.__index_tools.get_federated_indexes()
        return self.__federated_indexes

    async def get_tools(self) -> List[Tool]:
        if not self.enabled:
            return []

        tools: List[Tool] = [
            Tool.from_callable(
                name="web_fetch",
                executable=self.web_fetch,
                description=_WEB_FETCH_DESCRIPTION,
            ),
        ]

        indexes = await self.__get_federated_indexes()
        if indexes:
            engine_descriptions = ", ".join(
                f'"{idx.get("index_id")}"' f' ({idx.get("title", idx.get("index_id"))})'
                for idx in indexes
            )
            tools.append(
                Tool.from_callable(
                    name="web_search",
                    executable=self.web_search,
                    description=(
                        "Search the web or external sources using a"
                        " federated search engine.\n\n"
                        f"Available engines: {engine_descriptions}\n\n"
                        "Args:\n"
                        "    query: The search query.\n"
                        "    engine: The engine identifier to use"
                        f" (one of the available engines above).\n"
                        "    page: Result page number (default: 1).\n"
                        "    urls: Optional list of URLs to restrict"
                        " results to specific sites."
                    ),
                ),
            )

        return tools

    @streamable(
        running_text="Fetching web page {{ url }}...",
        completed_text="Fetched {{ url }}.",
        params_transform=hide,
        response_transform=hide,
    )
    async def web_fetch(self, url: str, prompt: Optional[str] = None) -> str:
        """Fetch a web page and return its content.

        Args:
            url: The URL to fetch.
            prompt: Optional extraction prompt — when provided, an LLM
                filters the page content to only the parts relevant to
                this prompt.
        """
        result = await self.__crawler.safe_crawl(url, prompt=prompt)
        if result is None:
            return f"Failed to fetch content from {url}."
        if not result.strip():
            return f"The page at {url} returned no readable content."
        return result

    @streamable(
        running_text="Searching the web for '{{ query }}'...",
        completed_text=(
            "Found {{ results|length }} web result"
            "{{ 's' if results|length != 1 else '' }} about '{{ query }}'."
        ),
        params_transform=hide,
        response_transform=hide,
    )
    async def web_search(
        self,
        query: str,
        engine: str,
        page: int = 1,
        urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search the web using a federated search engine.

        Args:
            query: The search query.
            engine: The engine identifier (index_id of a federated index).
            page: Result page number (default: 1).
            urls: Optional list of URLs to restrict search to
                (adds site: filters to the query).

        Returns:
            Dict with 'results', 'total_hits', and 'page'.
        """
        return await self.__index_tools.search_federated(
            query=query,
            index_id=engine,
            page=page,
            urls=urls,
        )

    async def to_prompt(self) -> str:
        if not self.enabled:
            return ""
        parts = [_SYSTEM_PROMPT_SECTION]
        indexes = await self.__get_federated_indexes()
        if indexes:
            engine_list = "\n".join(
                f"- {idx.get('index_id')}:" f" {idx.get('title', idx.get('index_id'))}"
                for idx in indexes
            )
            parts.append(
                "\n## Web search\n\n"
                "You have access to a `web_search` tool"
                " for searching external sources.\n"
                f"Available search engines:\n{engine_list}"
            )
        return "\n".join(parts)


class WebToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        index_tools: IndexToolsSource,
        web_tools_source_configuration: WebToolsSourceConfiguration = (
            WebToolsSourceConfiguration()
        ),
    ) -> WebToolsSource:
        crawler = WebPageCrawler(
            output_markdown=web_tools_source_configuration.output_markdown,
            max_content_chars=web_tools_source_configuration.max_content_chars,
            timeout_seconds=web_tools_source_configuration.timeout_seconds,
            cache_max_size=web_tools_source_configuration.cache_max_size,
            cache_ttl_seconds=web_tools_source_configuration.cache_ttl_seconds,
            upgrade_to_https=web_tools_source_configuration.upgrade_to_https,
            blocked_hostnames=web_tools_source_configuration.blocked_hostnames,
        )
        return WebToolsSource(
            crawler=crawler,
            enabled=web_tools_source_configuration.enabled,
            index_tools=index_tools,
        )
