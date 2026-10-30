from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from zav.agents_sdk.domain.agent_dependency import DependencyGroup


class SystemPromptSource(ABC):
    """Text added to the system prompt after the agent's ``system_prompt``."""

    source_name: ClassVar[str]
    # One line for admins choosing sources, e.g. in the agent editor.
    source_description: ClassVar[Optional[str]] = None
    # The tenant gate that hides this source until a tenant enables it.
    source_gate: ClassVar[Optional[str]] = None
    enabled: bool

    @abstractmethod
    async def to_prompt(self) -> str:
        raise NotImplementedError


class SystemPromptSourceGroup(DependencyGroup[SystemPromptSource]):
    __collects__ = SystemPromptSource
