from abc import ABC, abstractmethod
from typing import ClassVar

from zav.agents_sdk.domain.agent_dependency import DependencyGroup


class InstructionSource(ABC):
    """Abstract source for system-level instructions.

    Extend this to inject cross-cutting behavioral instructions into
    the agent's system prompt — e.g. citation formatting rules,
    response-style guidelines, or domain-specific conventions that are
    not tied to any single tool or skill.
    """

    source_name: ClassVar[str]
    enabled: bool

    @abstractmethod
    async def to_prompt(self) -> str:
        raise NotImplementedError


class InstructionSourceGroup(DependencyGroup[InstructionSource]):
    __collects__ = InstructionSource
