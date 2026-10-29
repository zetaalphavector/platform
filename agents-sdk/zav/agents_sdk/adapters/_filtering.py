from typing import ClassVar, List, Optional, Protocol, Sequence, Set, TypeVar


class _Named(Protocol):
    source_name: ClassVar[str]
    enabled: bool


TNamed = TypeVar("TNamed", bound=_Named)


def active_sources(
    sources: Sequence[TNamed],
    enabled: bool,
    include: Optional[Set[str]],
    exclude: Optional[Set[str]],
) -> List[TNamed]:
    """The sources a provider works with: none when the provider is off, else
    every source that passes ``is_source_active``."""
    if not enabled:
        return []
    return [
        source
        for source in sources
        if is_source_active(source.source_name, source.enabled, include, exclude)
    ]


def pick_writable(
    candidates: Sequence[TNamed], name: Optional[str]
) -> Optional[TNamed]:
    """The one source writes go to: the named one if it is among the candidates,
    else the first candidate. None when nothing qualifies."""
    if name is not None:
        return next((c for c in candidates if c.source_name == name), None)
    return candidates[0] if candidates else None


def is_source_active(
    name: str,
    source_enabled: bool,
    include: Optional[Set[str]],
    exclude: Optional[Set[str]],
) -> bool:
    """Determine whether a source should be active using valve-chain logic.

    Evaluation order:
    1. ``exclude`` — hard veto; if the name is in the exclude set the source
       is always OFF regardless of other settings.
    2. ``include`` — if an include allowlist is defined and the name is in it
       the source is ON (overrides ``source_enabled``).  If the allowlist is
       defined but the name is *not* in it the source is OFF.
    3. ``source_enabled`` — when no include allowlist is defined, the source's
       own enabled flag is the final gate.
    """
    if exclude is not None and name in exclude:
        return False
    if include is not None:
        return name in include
    return source_enabled


def passes_name_filter(
    name: str,
    include: Optional[Set[str]],
    exclude: Optional[Set[str]],
) -> bool:
    if include is not None and name not in include:
        return False
    if exclude is not None and name in exclude:
        return False
    return True
