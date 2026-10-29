import asyncio
import ipaddress
import socket
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urlunparse

import httpx

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
ResolveHost = Callable[[str], Awaitable[list[IPAddress]]]
ResolveHostSync = Callable[[str], list[IPAddress]]

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
_LOCALHOST_NAMES = frozenset({"localhost", "localhost.localdomain"})

# Canonical SSRF denylists shared by every caller of validate_url.
BLOCKED_HOSTNAMES = frozenset({"metadata.google.internal"})

BLOCKED_SUFFIXES = (
    ".internal",
    ".local",
    ".svc",
    ".svc.cluster.local",
    ".pod.cluster.local",
)

OAST_SUFFIXES = (
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


class CrawlBlocked(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ParsedUrl:
    scheme: str
    hostname: str
    port: int
    url: str
    is_ip_literal: bool


def _ip_is_blocked(addr: IPAddress) -> bool:
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    ):
        return True
    if isinstance(addr, ipaddress.IPv4Address):
        # CGNAT (100.64.0.0/10) and 0.0.0.0/8 are not flagged as private on
        # every Python version; check explicitly.
        if addr in ipaddress.ip_network("100.64.0.0/10"):
            return True
        if addr in ipaddress.ip_network("0.0.0.0/8"):
            return True
    return False


def _parse_ip_literal(host: str) -> IPAddress | None:
    if ":" in host:
        try:
            return ipaddress.IPv6Address(host)
        except ValueError:
            return None
    try:
        # socket.inet_aton accepts the legacy octal/decimal/hex and short-form
        # IPv4 encodings (e.g. "0177.0.0.1", "2130706433", "0x7f.1") that
        # ipaddress.IPv4Address rejects. Normalizing through it closes the
        # encoding-bypass surface.
        packed = socket.inet_aton(host)
    except OSError:
        return None
    return ipaddress.IPv4Address(packed)


def _matches_suffix(host: str, suffixes: Sequence[str]) -> bool:
    return any(host == s.lstrip(".") or host.endswith(s) for s in suffixes)


def validate_url(
    url: str,
    *,
    allowed_ports: set[int],
    blocked_exact_hostnames: frozenset[str] = frozenset(),
    blocked_suffixes: Sequence[str] = (),
    oast_suffixes: Sequence[str] = (),
    allowed_hostnames: Optional[Sequence[str]] = None,
) -> ParsedUrl:
    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise CrawlBlocked("unsupported_scheme")

    if parsed.username is not None or parsed.password is not None:
        raise CrawlBlocked("userinfo_not_allowed")

    raw_host = parsed.hostname
    if not raw_host or any(c.isspace() for c in raw_host):
        raise CrawlBlocked("invalid_host")

    try:
        port = parsed.port if parsed.port is not None else _DEFAULT_PORTS[scheme]
    except ValueError:
        raise CrawlBlocked("invalid_port")
    if port not in allowed_ports:
        raise CrawlBlocked("port_not_allowed")

    ip_literal = _parse_ip_literal(raw_host)
    if ip_literal is not None:
        if _ip_is_blocked(ip_literal):
            raise CrawlBlocked("blocked_ip")
        host = raw_host.lower()
        is_ip_literal = True
    else:
        try:
            host = raw_host.encode("idna").decode("ascii").lower()
        except (UnicodeError, ValueError):
            raise CrawlBlocked("invalid_host")
        if (
            host in _LOCALHOST_NAMES
            or _matches_suffix(host, blocked_suffixes)
            or host in blocked_exact_hostnames
        ):
            raise CrawlBlocked("blocked_host")
        if _matches_suffix(host, oast_suffixes):
            raise CrawlBlocked("blocked_oast")
        if allowed_hostnames and host not in {h.lower() for h in allowed_hostnames}:
            raise CrawlBlocked("host_not_allowed")
        is_ip_literal = False

    netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    normalized_url = urlunparse(
        (scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )
    return ParsedUrl(
        scheme=scheme,
        hostname=host,
        port=port,
        url=normalized_url,
        is_ip_literal=is_ip_literal,
    )


def _addresses_from_getaddrinfo(infos: Sequence[tuple]) -> list[IPAddress]:
    return [ipaddress.ip_address(info[4][0]) for info in infos]


class _CachedResolver:
    def __init__(self, ttl_seconds: float = 30.0) -> None:
        self.__ttl = ttl_seconds
        self.__entries: dict[str, tuple[list[IPAddress], float]] = {}

    async def __call__(self, host: str) -> list[IPAddress]:
        cached = self.__entries.get(host)
        if cached is not None and time.monotonic() - cached[1] <= self.__ttl:
            return cached[0]
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        addresses = _addresses_from_getaddrinfo(infos)
        self.__entries[host] = (addresses, time.monotonic())
        return addresses


class _CachedResolverSync:
    def __init__(self, ttl_seconds: float = 30.0) -> None:
        self.__ttl = ttl_seconds
        self.__entries: dict[str, tuple[list[IPAddress], float]] = {}

    def __call__(self, host: str) -> list[IPAddress]:
        cached = self.__entries.get(host)
        if cached is not None and time.monotonic() - cached[1] <= self.__ttl:
            return cached[0]
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        addresses = _addresses_from_getaddrinfo(infos)
        self.__entries[host] = (addresses, time.monotonic())
        return addresses


resolve_host: ResolveHost = _CachedResolver()
resolve_host_sync: ResolveHostSync = _CachedResolverSync()


def addresses_are_safe(addresses: Sequence[IPAddress]) -> bool:
    return bool(addresses) and not any(_ip_is_blocked(addr) for addr in addresses)


DEFAULT_ALLOWED_PORTS = frozenset({80, 443})


@dataclass(frozen=True)
class UrlPolicy:
    allowed_ports: frozenset[int] = DEFAULT_ALLOWED_PORTS
    blocked_exact_hostnames: frozenset[str] = BLOCKED_HOSTNAMES
    blocked_suffixes: tuple[str, ...] = BLOCKED_SUFFIXES
    oast_suffixes: tuple[str, ...] = OAST_SUFFIXES
    allowed_hostnames: Optional[tuple[str, ...]] = None

    def validate(self, url: str) -> ParsedUrl:
        return validate_url(
            url,
            allowed_ports=set(self.allowed_ports),
            blocked_exact_hostnames=self.blocked_exact_hostnames,
            blocked_suffixes=self.blocked_suffixes,
            oast_suffixes=self.oast_suffixes,
            allowed_hostnames=self.allowed_hostnames,
        )


_DEFAULT_POLICY = UrlPolicy()


def validate_and_resolve(
    url: str, *, resolve_host: ResolveHostSync
) -> tuple[ParsedUrl, list[IPAddress]]:
    parsed = validate_url(
        url,
        allowed_ports=set(DEFAULT_ALLOWED_PORTS),
        blocked_exact_hostnames=BLOCKED_HOSTNAMES,
        blocked_suffixes=BLOCKED_SUFFIXES,
        oast_suffixes=OAST_SUFFIXES,
    )
    addresses: list[IPAddress] = []
    if not parsed.is_ip_literal:
        addresses = list(resolve_host(parsed.hostname))
        if not addresses_are_safe(addresses):
            raise CrawlBlocked("blocked_resolved_ip")
    return parsed, addresses


def pin_request_to_ip(request: httpx.Request, ip: str, host: str) -> None:
    if not ip or ip == host:
        # No pin to apply (IP literal or unresolved). Clear any SNI override a
        # previous redirect hop left on the inherited extensions dict so this
        # hop is not dialed with the wrong TLS hostname.
        if request.extensions.get("sni_hostname"):
            request.extensions = dict(request.extensions)
            request.extensions.pop("sni_hostname", None)
        return
    # Dial the validated IP while preserving the original Host header and pinning
    # TLS SNI to the original hostname. Closes the DNS-rebinding TOCTOU window
    # between resolution and connection.
    request.extensions = dict(request.extensions)
    request.extensions["sni_hostname"] = host
    request.url = request.url.copy_with(host=ip)


class SafeTransport(httpx.BaseTransport):
    """Validates, resolves and IP-pins every request (redirect hops included)
    before delegating to the wrapped transport. Raises CrawlBlocked for a URL
    that targets a disallowed or internal address. With resolve_and_pin=False
    the caller owns connection safety (e.g. an injected test transport), so only
    URL validation runs."""

    def __init__(
        self,
        *,
        resolve_host: ResolveHostSync,
        inner: Optional[httpx.BaseTransport] = None,
        policy: UrlPolicy = _DEFAULT_POLICY,
        resolve_and_pin: bool = True,
    ) -> None:
        self.__resolve_host = resolve_host
        self.__inner = inner or httpx.HTTPTransport()
        self.__policy = policy
        self.__resolve_and_pin = resolve_and_pin

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        original_url = request.url
        parsed = self.__policy.validate(str(request.url))
        ip = ""
        if self.__resolve_and_pin and not parsed.is_ip_literal:
            addresses = list(self.__resolve_host(parsed.hostname))
            if not addresses_are_safe(addresses):
                raise CrawlBlocked("blocked_resolved_ip")
            ip = str(addresses[0])
        pin_request_to_ip(request, ip, parsed.hostname)
        try:
            return self.__inner.handle_request(request)
        finally:
            # Restore the original host so httpx resolves the next redirect hop
            # against the real hostname, not the pinned IP.
            if ip:
                request.url = original_url

    def close(self) -> None:
        self.__inner.close()


class SafeAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        *,
        resolve_host: ResolveHost,
        inner: Optional[httpx.AsyncBaseTransport] = None,
        policy: UrlPolicy = _DEFAULT_POLICY,
        resolve_and_pin: bool = True,
    ) -> None:
        self.__resolve_host = resolve_host
        self.__inner = inner or httpx.AsyncHTTPTransport()
        self.__policy = policy
        self.__resolve_and_pin = resolve_and_pin

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        original_url = request.url
        parsed = self.__policy.validate(str(request.url))
        ip = ""
        if self.__resolve_and_pin and not parsed.is_ip_literal:
            addresses = list(await self.__resolve_host(parsed.hostname))
            if not addresses_are_safe(addresses):
                raise CrawlBlocked("blocked_resolved_ip")
            ip = str(addresses[0])
        pin_request_to_ip(request, ip, parsed.hostname)
        try:
            return await self.__inner.handle_async_request(request)
        finally:
            if ip:
                request.url = original_url

    async def aclose(self) -> None:
        await self.__inner.aclose()
