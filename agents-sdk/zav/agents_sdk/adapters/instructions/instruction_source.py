from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from zav.agents_sdk.domain.agent_dependency import DependencyGroup


class InstructionSource(ABC):
    """Abstract source for system-level instructions.

    Extend this to inject cross-cutting behavioral instructions into
    the agent's system prompt — e.g. citation formatting rules,
    response-style guidelines, or domain-specific conventions that are
    not tied to any single tool or skill.
    """

    source_name: ClassVar[str]
    # One line for admins choosing sources, e.g. in the agent editor.
    source_description: ClassVar[Optional[str]] = None
    # The tenant gate that hides this source until a tenant enables it.
    source_gate: ClassVar[Optional[str]] = None
    enabled: bool

    @abstractmethod
    async def to_prompt(self) -> str:
        raise NotImplementedError


class WritableInstructionSource(InstructionSource):
    """An instruction source the harness can add prompt notes to. Only sources
    of this type qualify as the instructions provider's ``writable`` source."""

    @abstractmethod
    async def write_note(self, name: str, content: str) -> None:
        """Create or replace the named instruction."""
        raise NotImplementedError


class InstructionSourceGroup(DependencyGroup[InstructionSource]):
    __collects__ = InstructionSource
