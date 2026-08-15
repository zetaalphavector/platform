import asyncio
import ipaddress
import socket
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urlunparse

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
ResolveHost = Callable[[str], Awaitable[list[IPAddress]]]

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
_LOCALHOST_NAMES = frozenset({"localhost", "localhost.localdomain"})


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
        addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
        self.__entries[host] = (addresses, time.monotonic())
        return addresses


resolve_host: ResolveHost = _CachedResolver()


def addresses_are_safe(addresses: Sequence[IPAddress]) -> bool:
    return bool(addresses) and not any(_ip_is_blocked(addr) for addr in addresses)
