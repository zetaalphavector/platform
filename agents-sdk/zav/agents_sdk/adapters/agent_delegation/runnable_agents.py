from typing import Optional, Set

from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class RunnableAgents:
    """The agents the caller may run. ``None`` places no restriction; a host
    that shares agents with some users registers its own factory."""

    def __init__(self, identifiers: Optional[Set[str]] = None):
        self.identifiers = identifiers

    def allows(self, agent_identifier: str) -> bool:
        return self.identifiers is None or agent_identifier in self.identifiers


class RunnableAgentsFactory(AgentDependencyFactory):
    @classmethod
    def create(cls) -> RunnableAgents:
        return RunnableAgents()
