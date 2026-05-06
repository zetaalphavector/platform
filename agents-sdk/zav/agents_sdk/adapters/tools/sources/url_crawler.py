import asyncio
import ipaddress
import re
import time
from html.parser import HTMLParser
from typing import Callable, Coroutine, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify
from zav.logging import logger

_MAX_IMAGE_SRC_LENGTH = 200


def _is_image_src_usable(src: Optional[str]) -> bool:
    if not src:
        return False
    if src.startswith("data:image/"):
        return False
    if src.startswith("blob:"):
        return False
    if len(src) > _MAX_IMAGE_SRC_LENGTH:
        return False
    if src.endswith(".svg"):
        return False
    return True


def _format_image_marker(attrs_dict: Dict[str, str]) -> Optional[str]:
    src = attrs_dict.get("src")
    if not _is_image_src_usable(src):
        return None
    parts = [f"Image src: {src}"]
    if alt := attrs_dict.get("alt"):
        parts.append(f"alt: {alt}")
    if title := attrs_dict.get("title"):
        parts.append(f"title: {title}")
    return "[" + " - ".join(parts) + "]"


class _HTMLToTextParser(HTMLParser):
    BLOCK_TAGS = frozenset(
        {
            "p",
            "div",
            "br",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "tr",
            "blockquote",
            "pre",
            "section",
            "article",
            "header",
            "footer",
            "nav",
            "table",
            "thead",
            "tbody",
            "hr",
        }
    )
    SKIP_TAGS = frozenset({"script", "style", "noscript"})

    def __init__(self, *, include_images: bool = False) -> None:
        super().__init__()
        self._pieces: List[str] = []
        self._skip = False
        self._include_images = include_images

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self.SKIP_TAGS:
            self._skip = True
        elif tag in self.BLOCK_TAGS:
            self._pieces.append("\n")
        elif self._include_images and tag == "img":
            marker = _format_image_marker(dict(attrs))
            if marker:
                self._pieces.append(f"\n{marker}\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            self._skip = False
        elif tag in self.BLOCK_TAGS:
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw = "".join(self._pieces)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


LLMExtractor = Callable[[str, str], Coroutine[None, None, str]]

_BLOCKED_HOSTNAMES = frozenset({"metadata.google.internal"})

_BLOCKED_SUFFIXES = (
    ".internal",
    ".local",
    ".svc",
    ".svc.cluster.local",
    ".pod.cluster.local",
)


def _is_private_url(
    url: str,
    extra_blocked_hostnames: frozenset = frozenset(),
) -> bool:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return True
    if parsed.port is not None:
        return True
    if hostname in _BLOCKED_HOSTNAMES or hostname in extra_blocked_hostnames:
        return True
    try:
        addr = ipaddress.ip_address(hostname)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        lower = hostname.lower()
        if lower in ("localhost", "localhost.localdomain"):
            return True
        return any(lower.endswith(suffix) for suffix in _BLOCKED_SUFFIXES)


class _TTLCache:
    __slots__ = ("__max_size", "__ttl", "__entries")

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self.__max_size = max_size
        self.__ttl = ttl_seconds
        self.__entries: Dict[str, tuple] = {}

    def get(self, key: str) -> Optional[str]:
        if self.__max_size <= 0:
            return None
        entry = self.__entries.get(key)
        if entry is None:
            return None
        content, created_at = entry
        if time.monotonic() - created_at > self.__ttl:
            del self.__entries[key]
            return None
        return content

    def put(self, key: str, content: str) -> None:
        if self.__max_size <= 0:
            return
        self.__evict_expired()
        if len(self.__entries) >= self.__max_size and key not in self.__entries:
            oldest = min(self.__entries, key=lambda k: self.__entries[k][1])
            del self.__entries[oldest]
        self.__entries[key] = (content, time.monotonic())

    def __evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, ts) in self.__entries.items() if now - ts > self.__ttl]
        for k in expired:
            del self.__entries[k]


_REMOVE_TAGS = ["script", "style", "noscript", "link"]


def _is_xml_content_type(content_type: str) -> bool:
    ct = content_type.lower().split(";")[0].strip()
    return ct in ("application/xml", "text/xml", "application/atom+xml")


def _strip_boilerplate(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_REMOVE_TAGS):
        tag.decompose()
    return str(soup)


def parse_html(
    html: str,
    *,
    as_markdown: bool = False,
    is_xml: bool = False,
    include_images: bool = False,
) -> str:
    if is_xml:
        # XML feeds (e.g. Atom/RSS) have no HTML boilerplate; skip
        # BeautifulSoup to avoid XMLParsedAsHTMLWarning and the lxml dep.
        parser = _HTMLToTextParser()
        parser.feed(html)
        return parser.get_text()
    if as_markdown:
        return markdownify(_strip_boilerplate(html), heading_style="ATX").strip()
    parser = _HTMLToTextParser(include_images=include_images)
    parser.feed(html)
    return parser.get_text()


class WebPageCrawler:
    def __init__(
        self,
        *,
        headers: Optional[Dict[str, str]] = None,
        max_content_chars: Optional[int] = None,
        output_markdown: bool = False,
        include_images: bool = False,
        llm_extractor: Optional[LLMExtractor] = None,
        cache_max_size: int = 0,
        cache_ttl_seconds: float = 900.0,
        timeout_seconds: Optional[float] = None,
        upgrade_to_https: bool = False,
        http_client: Optional[httpx.AsyncClient] = None,
        blocked_hostnames: Optional[List[str]] = None,
    ) -> None:
        self.__headers = headers
        self.__max_content_chars = max_content_chars
        self.__output_markdown = output_markdown
        self.__include_images = include_images
        self.__llm_extractor = llm_extractor
        self.__timeout_seconds = timeout_seconds
        self.__upgrade_to_https = upgrade_to_https
        self.__cache = _TTLCache(cache_max_size, cache_ttl_seconds)
        self.__http_client = http_client
        self.__blocked_hostnames = (
            frozenset(blocked_hostnames) if blocked_hostnames else frozenset()
        )

    async def crawl(
        self,
        url: str,
        *,
        prompt: Optional[str] = None,
    ) -> str:
        url = self.__prepare_url(url)

        cache_key = f"{url}|{prompt or ''}"
        if cached := self.__cache.get(cache_key):
            return cached

        try:
            content = await self.__fetch_and_parse(url)
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return ""

        if self.__max_content_chars:
            content = content[: self.__max_content_chars]

        if prompt and self.__llm_extractor:
            try:
                content = await self.__llm_extractor(content, prompt)
            except Exception as e:
                logger.warning(f"LLM extraction failed for {url}, returning raw: {e}")

        self.__cache.put(cache_key, content)
        return content

    async def safe_crawl(
        self,
        url: str,
        *,
        prompt: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> Optional[str]:
        effective_timeout = request_timeout or self.__timeout_seconds
        try:
            if effective_timeout:
                return await asyncio.wait_for(
                    self.crawl(url, prompt=prompt),
                    timeout=effective_timeout,
                )
            return await self.crawl(url, prompt=prompt)
        except asyncio.TimeoutError:
            logger.debug(f"Timeout crawling URL {url}")
            return None
        except Exception as e:
            logger.debug(f"Error crawling URL {url}: {e}")
            return None

    def __prepare_url(self, url: str) -> str:
        if self.__upgrade_to_https and url.startswith("http://"):
            url = "https://" + url[len("http://") :]
        if _is_private_url(url, self.__blocked_hostnames):
            raise ValueError(f"Requests to private/internal URLs are blocked: {url}")
        return url

    async def __fetch_and_parse(self, url: str) -> str:
        timeout = (
            httpx.Timeout(self.__timeout_seconds) if self.__timeout_seconds else None
        )
        if self.__http_client is not None:
            response = await self.__http_client.get(
                url, headers=self.__headers, timeout=timeout, follow_redirects=True
            )
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self.__headers,
                    timeout=timeout,
                    follow_redirects=True,
                )
        content_type = response.headers.get("content-type", "")
        is_xml = _is_xml_content_type(content_type)
        return parse_html(
            response.text,
            as_markdown=self.__output_markdown,
            is_xml=is_xml,
            include_images=self.__include_images,
        )
