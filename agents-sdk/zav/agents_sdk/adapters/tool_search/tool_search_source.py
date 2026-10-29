from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from zav.llm_domain import ToolSearchMethodSettings

from zav.agents_sdk.domain.agent_dependency import DependencyGroup


class ToolSearchSource(ABC):
    """One way to load deferred tools on demand, with its settings. The model
    configuration lists the methods a model supports; the provider offers its
    active sources in preference order and each model call uses the first one
    its model supports."""

    source_name: ClassVar[str]
    # One line for admins choosing sources, e.g. in the agent editor.
    source_description: ClassVar[Optional[str]] = None
    # The tenant gate that hides this source until a tenant enables it.
    source_gate: ClassVar[Optional[str]] = None
    enabled: bool

    @abstractmethod
    def settings(self) -> ToolSearchMethodSettings:
        raise NotImplementedError


class ToolSearchSourceGroup(DependencyGroup[ToolSearchSource]):
    __collects__ = ToolSearchSource
