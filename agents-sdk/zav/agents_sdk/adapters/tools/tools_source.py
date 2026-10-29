from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.agent_dependency import DependencyGroup
from zav.agents_sdk.domain.tools import Tool


class ToolsSourceConfiguration(BaseModel):
    defer_tools: Optional[bool] = Field(
        None,
        description=(
            "Load this source's tools on demand through tool search. Unset "
            "follows the tools provider's defer_tools."
        ),
    )


class ToolsSource(ABC):
    """Abstract source for tool discovery.

    Extend this to provide custom Python-defined tools to agents without
    modifying agent code or creating new agent dependencies."""

    source_name: ClassVar[str]
    """Short identifier for this source, used in configuration filtering."""
    # One line for admins choosing sources, e.g. in the agent editor.
    source_description: ClassVar[Optional[str]] = None
    # The tenant gate that hides this source until a tenant enables it.
    source_gate: ClassVar[Optional[str]] = None
    enabled: bool
    # The source's own defer_tools; None follows the tools provider's.
    defer_tools: Optional[bool] = None

    @abstractmethod
    async def get_tools(self) -> List[Tool]:
        """Return all tools provided by this source."""
        raise NotImplementedError

    async def to_prompt(self) -> str:
        return ""


class ToolsSourceGroup(DependencyGroup[ToolsSource]):
    """Collects all registered `ToolsSource` subclass instances."""

    __collects__ = ToolsSource
