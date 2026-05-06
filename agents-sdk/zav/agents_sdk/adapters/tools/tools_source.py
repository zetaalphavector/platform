from abc import ABC, abstractmethod
from typing import ClassVar, List

from zav.agents_sdk.domain.agent_dependency import DependencyGroup
from zav.agents_sdk.domain.tools import Tool


class ToolsSource(ABC):
    """Abstract source for tool discovery.

    Extend this to provide custom Python-defined tools to agents without
    modifying agent code or creating new agent dependencies."""

    source_name: ClassVar[str]
    """Short identifier for this source, used in configuration filtering."""
    enabled: bool

    @abstractmethod
    async def get_tools(self) -> List[Tool]:
        """Return all tools provided by this source."""
        raise NotImplementedError

    async def to_prompt(self) -> str:
        return ""


class ToolsSourceGroup(DependencyGroup[ToolsSource]):
    """Collects all registered `ToolsSource` subclass instances."""

    __collects__ = ToolsSource
