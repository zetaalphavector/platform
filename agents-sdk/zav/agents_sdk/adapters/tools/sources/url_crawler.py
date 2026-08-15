import asyncio
import re
import time
from html.parser import HTMLParser
from typing import Callable, Coroutine, Optional

import httpx
import pymupdf
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify
from zav.logging import logger

from zav.agents_sdk.adapters.tools.sources import url_policy
from zav.agents_sdk.adapters.tools.sources.url_policy import CrawlBlocked, ResolveHost

_MAX_IMAGE_SRC_LENGTH = 200
_DEFAULT_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_DEFAULT_MAX_PDF_BYTES = 20 * 1024 * 1024
_DEFAULT_MAX_PDF_PAGES = 50
_DEFAULT_MAX_PDF_CHARS = 200_000
_MAX_REDIRECT_HOPS = 5
_FETCHED_CONTENT_TAG_RE = re.compile(r"</?fetched_content", re.IGNORECASE)

_REASON_TO_OUTCOME = {
    "blocked_oast": "blocked_oast",
    "redirect_without_location": "blocked_redirect",
    "too_many_redirects": "blocked_redirect",
    "unsupported_type": "blocked_type",
    "body_too_large": "blocked_size",
}


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


def _format_image_marker(attrs_dict: dict[str, str]) -> Optional[str]:
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
    SKIP_TAGS = frozenset(
        {
            "script",
            "style",
            "noscript",
            "iframe",
            "frame",
            "object",
            "embed",
            "svg",
            "math",
            "template",
            "form",
        }
    )

    def __init__(self, *, include_images: bool = False) -> None:
        super().__init__()
        self._pieces: list[str] = []
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

_OAST_DENYLIST = (
    ".oastify.com",
    ".burpcollaborator.net",
    ".interact.sh",
    ".oast.fun",
    ".oast.live",
    ".oast.pro",
    ".oast.me",
    ".oast.site",
    ".oast.online",
    ".webhook.site",
    ".requestbin.com",
    ".requestbin.net",
    ".pipedream.net",
    ".ngrok.io",
    ".ngrok-free.app",
    ".ngrok.app",
    ".ngrok.dev",
    ".loca.lt",
    ".serveo.net",
    ".trycloudflare.com",
    ".lhr.life",
)


class _TTLCache:
    __slots__ = ("__max_size", "__ttl", "__entries")

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self.__max_size = max_size
        self.__ttl = ttl_seconds
        self.__entries: dict[str, tuple] = {}

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


_REMOVE_TAGS = [
    "script",
    "style",
    "noscript",
    "link",
    "meta",
    "iframe",
    "frame",
    "object",
    "embed",
    "svg",
    "math",
    "template",
    "form",
    "input",
]
_DROP_ATTRS = frozenset({"style", "srcdoc"})
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})

_DEFAULT_ALLOWED_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "text/plain",
        "application/xml",
        "text/xml",
        "application/atom+xml",
        "application/rss+xml",
        "application/pdf",
    }
)

_XML_CONTENT_TYPES = frozenset(
    {
        "application/xml",
        "text/xml",
        "application/atom+xml",
        "application/rss+xml",
    }
)

_EXTENSION_CONTENT_TYPES = {
    ".html": "text/html",
    ".htm": "text/html",
    ".xhtml": "application/xhtml+xml",
    ".txt": "text/plain",
    ".xml": "application/xml",
    ".atom": "application/atom+xml",
    ".rss": "application/rss+xml",
    ".pdf": "application/pdf",
}


def _sniff_content_type(body: bytes) -> Optional[str]:
    head = body[:1024].lstrip()
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    lowered = head[:512].lower()
    if lowered.startswith(b"<?xml"):
        return "application/xml"
    if lowered.startswith(b"<!doctype html") or lowered.startswith(b"<html"):
        return "text/html"
    return None


def _extension_content_type(url: str) -> Optional[str]:
    path = httpx.URL(url).path.lower()
    for ext, media_type in _EXTENSION_CONTENT_TYPES.items():
        if path.endswith(ext):
            return media_type
    return None


def _resolve_content_type(content_type: str, body: bytes, url: str) -> Optional[str]:
    declared = content_type.split(";")[0].strip().lower() or None
    sniffed = _sniff_content_type(body)
    # Magic bytes win over the declared type for PDF/HTML so a PDF mislabelled
    # as HTML (or vice versa) is routed by its real bytes.
    if sniffed in ("application/pdf", "text/html") and declared in (
        "application/pdf",
        "text/html",
    ):
        return sniffed
    if declared:
        return declared
    if sniffed:
        return sniffed
    return _extension_content_type(url)


def _apply_ip_pin(request: httpx.Request, host_to_ip: dict[str, str]) -> None:
    host = request.url.host
    ip = host_to_ip.get(host)
    if not ip or ip == host:
        return
    # Preserve the original Host header and
    # pin TLS SNI / certificate verification to the original hostname, while
    # dialing the validated IP. This closes the DNS-rebinding TOCTOU window
    # between resolution and connection.
    request.extensions = dict(request.extensions)
    request.extensions["sni_hostname"] = host
    request.url = request.url.copy_with(host=ip)


class _IPPinnedTransport(httpx.AsyncHTTPTransport):
    def __init__(self, host_to_ip: dict[str, str]) -> None:
        super().__init__()
        self.__host_to_ip = host_to_ip

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        _apply_ip_pin(request, self.__host_to_ip)
        return await super().handle_async_request(request)


def _sanitize_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    for tag in soup.find_all(_REMOVE_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            lowered = attr.lower()
            if lowered.startswith("on") or lowered in _DROP_ATTRS:
                del tag.attrs[attr]
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
        return markdownify(_sanitize_html(html), heading_style="ATX").strip()
    parser = _HTMLToTextParser(include_images=include_images)
    parser.feed(html)
    return parser.get_text()


class WebPageCrawler:
    def __init__(
        self,
        *,
        headers: Optional[dict[str, str]] = None,
        max_content_chars: Optional[int] = None,
        output_markdown: bool = False,
        include_images: bool = False,
        llm_extractor: Optional[LLMExtractor] = None,
        cache_max_size: int = 0,
        cache_ttl_seconds: float = 900.0,
        timeout_seconds: Optional[float] = None,
        upgrade_to_https: bool = False,
        http_client: Optional[httpx.AsyncClient] = None,
        blocked_hostnames: Optional[list[str]] = None,
        allowed_ports: Optional[set[int]] = None,
        allowed_hostnames: Optional[list[str]] = None,
        resolve_host: Optional[ResolveHost] = None,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
        allowed_content_types: Optional[list[str]] = None,
        max_pdf_bytes: int = _DEFAULT_MAX_PDF_BYTES,
        max_pdf_pages: int = _DEFAULT_MAX_PDF_PAGES,
        max_pdf_chars: int = _DEFAULT_MAX_PDF_CHARS,
        untrusted_content_envelope: bool = True,
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
        self.__blocked_suffixes = tuple(blocked_hostnames) if blocked_hostnames else ()
        self.__allowed_ports: set[int] = allowed_ports or {80, 443}

        self.__allowed_hostnames = allowed_hostnames
        self.__resolve_host = resolve_host or url_policy.resolve_host
        self.__max_response_bytes = max_response_bytes
        self.__allowed_content_types: frozenset[str] = (
            frozenset(ct.lower() for ct in allowed_content_types)
            if allowed_content_types is not None
            else _DEFAULT_ALLOWED_CONTENT_TYPES
        )
        self.__max_pdf_bytes = max_pdf_bytes
        self.__max_pdf_pages = max_pdf_pages
        self.__max_pdf_chars = max_pdf_chars
        self.__untrusted_content_envelope = untrusted_content_envelope

    async def crawl(
        self,
        url: str,
        *,
        prompt: Optional[str] = None,
    ) -> str:
        started = time.monotonic()
        try:
            url = await self.__prepare_url(url)

            cache_key = f"{url}|{prompt or ''}"
            if cached := self.__cache.get(cache_key):
                return cached

            final_url, status, content_type, body = await self.__fetch(url)
            content, resolved_type, truncated = await self.__decode_and_parse(
                final_url, content_type, body
            )
        except CrawlBlocked as e:
            self.__log_outcome(
                _REASON_TO_OUTCOME.get(e.reason, "blocked_ssrf"),
                url=url,
                block_reason=e.reason,
                duration_ms=self.__elapsed_ms(started),
            )
            raise
        except Exception as e:
            self.__log_outcome(
                "parser_error",
                url=url,
                block_reason=type(e).__name__,
                duration_ms=self.__elapsed_ms(started),
            )
            raise

        if self.__max_content_chars and len(content) > self.__max_content_chars:
            content = content[: self.__max_content_chars]
            truncated = True

        if prompt and self.__llm_extractor:
            try:
                content = await self.__llm_extractor(content, prompt)
            except Exception as e:
                logger.warning(f"LLM extraction failed for {url}, returning raw: {e}")

        if self.__untrusted_content_envelope:
            content = self.__envelope_wrap(
                content,
                url=final_url,
                content_type=resolved_type,
                truncated=truncated,
            )

        self.__log_outcome(
            "fetched",
            url=url,
            final_url=final_url,
            status=status,
            content_type=resolved_type,
            bytes_read=len(body),
            truncated=truncated,
            duration_ms=self.__elapsed_ms(started),
        )
        self.__cache.put(cache_key, content)
        return content

    @staticmethod
    def __elapsed_ms(started: float) -> int:
        return int((time.monotonic() - started) * 1000)

    def __log_outcome(
        self,
        outcome: str,
        *,
        url: str,
        final_url: Optional[str] = None,
        status: Optional[int] = None,
        content_type: Optional[str] = None,
        bytes_read: int = 0,
        truncated: bool = False,
        block_reason: Optional[str] = None,
        duration_ms: int = 0,
    ) -> None:
        logger.info(
            "web_fetch",
            extra={
                "outcome": outcome,
                "url": url,
                "final_url": final_url,
                "status": status,
                "content_type": content_type,
                "bytes_read": bytes_read,
                "truncated": truncated,
                "block_reason": block_reason,
                "duration_ms": duration_ms,
            },
        )

    async def safe_crawl(
        self,
        url: str,
        *,
        prompt: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> Optional[str]:
        effective_timeout = request_timeout or self.__timeout_seconds
        started = time.monotonic()
        try:
            if effective_timeout:
                return await asyncio.wait_for(
                    self.crawl(url, prompt=prompt),
                    timeout=effective_timeout,
                )
            return await self.crawl(url, prompt=prompt)
        except asyncio.TimeoutError:
            self.__log_outcome(
                "timeout",
                url=url,
                block_reason="timeout",
                duration_ms=self.__elapsed_ms(started),
            )
            return None
        except Exception as e:
            logger.debug(f"Error crawling URL {url}: {e}")
            return None

    async def __prepare_url(self, url: str) -> str:
        if self.__upgrade_to_https and url.startswith("http://"):
            url = "https://" + url[len("http://") :]
        parsed, _ = await self.__validate_and_resolve(url)
        return parsed.url

    async def __validate_and_resolve(
        self, url: str
    ) -> tuple[url_policy.ParsedUrl, list[url_policy.IPAddress]]:
        parsed = url_policy.validate_url(
            url,
            allowed_ports=self.__allowed_ports,
            blocked_exact_hostnames=_BLOCKED_HOSTNAMES,
            blocked_suffixes=_BLOCKED_SUFFIXES + self.__blocked_suffixes,
            oast_suffixes=_OAST_DENYLIST,
            allowed_hostnames=self.__allowed_hostnames,
        )
        # When an http_client is injected the caller owns connection safety, so
        # DNS-resolution validation and IP pinning are skipped.
        addresses: list[url_policy.IPAddress] = []
        if self.__http_client is None and not parsed.is_ip_literal:
            addresses = list(await self.__resolve_host(parsed.hostname))
            if not url_policy.addresses_are_safe(addresses):
                raise CrawlBlocked("blocked_resolved_ip")
        return parsed, addresses

    async def __fetch(self, url: str) -> tuple[str, int, str, bytes]:
        timeout = (
            httpx.Timeout(self.__timeout_seconds) if self.__timeout_seconds else None
        )
        if self.__http_client is not None:
            return await self.__follow_and_read(self.__http_client, url, timeout)
        host_to_ip: dict[str, str] = {}
        transport = _IPPinnedTransport(host_to_ip)
        async with httpx.AsyncClient(transport=transport) as client:
            return await self.__follow_and_read(
                client, url, timeout, host_to_ip=host_to_ip
            )

    async def __follow_and_read(
        self,
        client: httpx.AsyncClient,
        url: str,
        timeout: Optional[httpx.Timeout],
        *,
        host_to_ip: Optional[dict[str, str]] = None,
    ) -> tuple[str, int, str, bytes]:
        current = url
        for _ in range(_MAX_REDIRECT_HOPS + 1):
            parsed, addresses = await self.__validate_and_resolve(current)
            if host_to_ip is not None and addresses:
                host_to_ip[parsed.hostname] = str(addresses[0])
            async with client.stream(
                "GET",
                parsed.url,
                headers=self.__headers,
                timeout=timeout,
                follow_redirects=False,
            ) as response:
                if response.status_code in _REDIRECT_STATUS_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise CrawlBlocked("redirect_without_location")
                    current = str(httpx.URL(parsed.url).join(location))
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                cap = self.__size_cap_for(content_type, str(parsed.url))
                body = await self.__read_body(response, cap)
                return str(parsed.url), response.status_code, content_type, body
        raise CrawlBlocked("too_many_redirects")

    def __size_cap_for(self, content_type: str, url: str) -> int:
        declared = content_type.split(";")[0].strip().lower()
        if declared == "application/pdf" or _extension_content_type(url) == (
            "application/pdf"
        ):
            return self.__max_pdf_bytes
        return self.__max_response_bytes

    async def __read_body(self, response: httpx.Response, cap: int) -> bytes:
        declared = response.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > cap:
                    raise CrawlBlocked("body_too_large")
            except ValueError:
                pass
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > cap:
                raise CrawlBlocked("body_too_large")
            chunks.append(chunk)
        return b"".join(chunks)

    async def __decode_and_parse(
        self, url: str, content_type: str, body: bytes
    ) -> tuple[str, str, bool]:
        resolved_type = _resolve_content_type(content_type, body, url)
        if resolved_type is None or resolved_type not in self.__allowed_content_types:
            raise CrawlBlocked("unsupported_type")
        if resolved_type == "application/pdf":
            text, truncated = await self.__parse_pdf(body)
            return text, resolved_type, truncated
        charset = "utf-8"
        for part in content_type.split(";")[1:]:
            key, _, value = part.strip().partition("=")
            if key.strip().lower() == "charset" and value:
                charset = value.strip().strip('"').strip("'")
        try:
            text = body.decode(charset, errors="replace")
        except LookupError:
            text = body.decode("utf-8", errors="replace")
        if resolved_type == "text/plain":
            return text, resolved_type, False
        parsed = parse_html(
            text,
            as_markdown=self.__output_markdown,
            is_xml=resolved_type in _XML_CONTENT_TYPES,
            include_images=self.__include_images,
        )
        return parsed, resolved_type, False

    async def __parse_pdf(self, body: bytes) -> tuple[str, bool]:
        if len(body) > self.__max_pdf_bytes:
            raise CrawlBlocked("body_too_large")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.__extract_pdf_text, body)

    def __extract_pdf_text(self, body: bytes) -> tuple[str, bool]:
        pieces: list[str] = []
        total = 0
        with pymupdf.open(stream=body, filetype="pdf") as doc:
            page_limit = min(doc.page_count, self.__max_pdf_pages)
            truncated = doc.page_count > page_limit
            for index in range(page_limit):
                page = doc[index]
                text = page.get_text("text")  # type: ignore
                remaining = self.__max_pdf_chars - total
                if len(text) >= remaining:
                    pieces.append(text[:remaining])
                    truncated = True
                    break
                pieces.append(text)
                total += len(text)
        return "\n\n".join(pieces), truncated

    def __envelope_wrap(
        self, text: str, *, url: str, content_type: str, truncated: bool
    ) -> str:
        escaped = _FETCHED_CONTENT_TAG_RE.sub(
            lambda match: "&lt;" + match.group(0)[1:], text
        )
        escaped_url = _FETCHED_CONTENT_TAG_RE.sub(
            lambda match: "&lt;" + match.group(0)[1:], url
        )
        escaped_type = _FETCHED_CONTENT_TAG_RE.sub(
            lambda match: "&lt;" + match.group(0)[1:], content_type
        )
        return (
            f'<fetched_content url="{escaped_url}" content_type="{escaped_type}" '
            f'truncated="{str(truncated).lower()}">\n'
            f"{escaped}\n"
            "</fetched_content>"
        )
