from typing import Optional, Set


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
