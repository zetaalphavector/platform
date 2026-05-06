from typing import Callable, Coroutine, Dict, List, Optional

import httpx
from defusedxml import ElementTree
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.tools.sources.url_crawler import WebPageCrawler
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable

SitemapFetcher = Callable[[str], Coroutine[None, None, str]]

_SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

_LIST_PAGES_DESCRIPTION = """\
List all available platform documentation pages. Returns the page paths \
and last-modified dates from the documentation sitemap.

Use this tool to discover which documentation pages exist before reading \
a specific one. Look at the paths to decide which page is most relevant \
to the user's question.\
"""

_READ_PAGE_DESCRIPTION = """\
Read a platform documentation page and return its content as text.

Args:
    path: The page path from the documentation site \
(e.g. "/gen-ai/customize/getting-started"). Use `list_platform_doc_pages` \
first to discover available paths.\
"""

_SYSTEM_PROMPT_SECTION = """\
## Platform documentation

You have access to the platform's own documentation via \
`list_platform_doc_pages` and `read_platform_doc` tools.

- When a user asks how to do something on the platform, use these tools \
to find the answer in the official docs.
- First call `list_platform_doc_pages` to see what pages are available, \
then `read_platform_doc` with the most relevant path.
- Always prefer official documentation over your general knowledge for \
platform-specific questions.
- If the docs don't cover the user's question, say so honestly.\
"""


class PlatformDocsToolsSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable platform documentation tools.")
    docs_base_url: str = Field(
        "https://docs.zeta-alpha.com",
        description="Base URL of the documentation site.",
    )
    sitemap_path: str = Field(
        "/sitemap.xml",
        description="Path to the sitemap XML relative to docs_base_url.",
    )
    max_content_chars: Optional[int] = Field(
        None,
        description="Truncate fetched page content to this many characters.",
    )
    cache_max_size: int = Field(
        50,
        description="Maximum number of pages to cache.",
    )
    cache_ttl_seconds: float = Field(
        900.0,
        description="Cache TTL for fetched pages in seconds.",
    )
    sitemap_cache_ttl_seconds: float = Field(
        3600.0,
        description="How long to cache the parsed sitemap in seconds.",
    )
    timeout_seconds: float = Field(
        30.0,
        description="Timeout for HTTP requests.",
    )


class _SitemapEntry:
    __slots__ = ("path", "lastmod")

    def __init__(self, path: str, lastmod: Optional[str]) -> None:
        self.path = path
        self.lastmod = lastmod


async def _default_sitemap_fetcher(
    url: str,
    timeout_seconds: float = 30.0,
) -> str:
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
    return response.text


def _parse_sitemap(xml_text: str, base_url: str) -> List[_SitemapEntry]:
    entries: List[_SitemapEntry] = []
    root = ElementTree.fromstring(xml_text)
    for url_el in root.findall("sm:url", _SITEMAP_NS):
        loc_el = url_el.find("sm:loc", _SITEMAP_NS)
        if loc_el is None or not loc_el.text:
            continue
        loc = loc_el.text.strip()
        path = loc.removeprefix(base_url)
        if not path.startswith("/"):
            path = "/" + path

        lastmod_el = url_el.find("sm:lastmod", _SITEMAP_NS)
        lastmod = (
            lastmod_el.text.strip()
            if lastmod_el is not None and lastmod_el.text
            else None
        )
        entries.append(_SitemapEntry(path=path, lastmod=lastmod))
    return entries


class PlatformDocsToolsSource(ToolsSource):

    source_name = "platform_docs"

    def __init__(
        self,
        crawler: WebPageCrawler,
        docs_base_url: str,
        sitemap_path: str,
        timeout_seconds: float,
        enabled: bool,
        sitemap_fetcher: Optional[SitemapFetcher] = None,
    ) -> None:
        self.__crawler = crawler
        self.__docs_base_url = docs_base_url.rstrip("/")
        self.__sitemap_url = self.__docs_base_url + sitemap_path
        self.__timeout_seconds = timeout_seconds
        self.enabled = enabled
        self.__sitemap_fetcher = sitemap_fetcher
        self.__sitemap: Optional[List[_SitemapEntry]] = None

    async def __fetch_sitemap(self) -> List[_SitemapEntry]:
        if self.__sitemap is not None:
            return self.__sitemap

        entries: List[_SitemapEntry] = []
        try:
            fetcher = self.__sitemap_fetcher
            if fetcher is not None:
                xml_text = await fetcher(self.__sitemap_url)
            else:
                xml_text = await _default_sitemap_fetcher(
                    self.__sitemap_url,
                    self.__timeout_seconds,
                )
            entries = _parse_sitemap(xml_text, self.__docs_base_url)
        except Exception as e:
            logger.error(f"Failed to fetch sitemap from" f" {self.__sitemap_url}: {e}")

        self.__sitemap = entries
        return entries

    async def get_tools(self) -> List[Tool]:
        if not self.enabled:
            return []
        return [
            Tool.from_callable(
                name="list_platform_doc_pages",
                executable=self.list_pages,
                description=_LIST_PAGES_DESCRIPTION,
            ),
            Tool.from_callable(
                name="read_platform_doc",
                executable=self.read_page,
                description=_READ_PAGE_DESCRIPTION,
            ),
        ]

    @streamable(
        running_text="Listing platform documentation pages...",
        completed_text="Found {{ entries|length }} documentation pages.",
        response_transform=hide,
    )
    async def list_pages(self) -> Dict[str, object]:
        entries = await self.__fetch_sitemap()
        return {
            "entries": [{"path": e.path, "last_modified": e.lastmod} for e in entries],
        }

    @streamable(
        running_text="Reading platform doc {{ path }}...",
        completed_text="Fetched documentation page from {{ path }}.",
        response_transform=hide,
    )
    async def read_page(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path

        url = self.__docs_base_url + path
        result = await self.__crawler.safe_crawl(
            url, request_timeout=self.__timeout_seconds
        )
        if result is None:
            return f"Failed to fetch documentation page at {path}."
        if not result.strip():
            return f"The documentation page at {path} returned no readable content."
        return result

    async def to_prompt(self) -> str:
        if not self.enabled:
            return ""
        return _SYSTEM_PROMPT_SECTION


class PlatformDocsToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        platform_docs_tools_source_configuration: PlatformDocsToolsSourceConfiguration = PlatformDocsToolsSourceConfiguration(),  # noqa: E501
    ) -> PlatformDocsToolsSource:
        cfg = platform_docs_tools_source_configuration
        crawler = WebPageCrawler(
            output_markdown=True,
            max_content_chars=cfg.max_content_chars,
            timeout_seconds=cfg.timeout_seconds,
            cache_max_size=cfg.cache_max_size,
            cache_ttl_seconds=cfg.cache_ttl_seconds,
        )
        return PlatformDocsToolsSource(
            crawler=crawler,
            docs_base_url=cfg.docs_base_url,
            sitemap_path=cfg.sitemap_path,
            timeout_seconds=cfg.timeout_seconds,
            enabled=cfg.enabled,
        )
