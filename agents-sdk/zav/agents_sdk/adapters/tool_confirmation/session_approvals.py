from typing import Any, Dict, Optional, Set

from zav.agents_sdk.domain.agent_dependency import (
    AgentDependencyFactory,
    ResumableAgentDependency,
)


class SessionApprovals(ResumableAgentDependency):
    """The tools the user allowed for the rest of this chat: they run without
    asking, unless a tool options source requires the ask."""

    state_key = "tool_confirmation"

    def __init__(self) -> None:
        self.__allowed: Set[str] = set()
        self.__dumped_state: Optional[Dict[str, Any]] = None

    async def dump(self) -> Dict[str, Any]:
        # Same object while unchanged, so the state saver skips the write.
        allowed = sorted(self.__allowed)
        if (
            self.__dumped_state is None
            or self.__dumped_state["session_allowed"] != allowed
        ):
            self.__dumped_state = {"session_allowed": allowed}
        return self.__dumped_state

    async def load(self, state: Dict[str, Any]) -> None:
        self.__allowed = set(state.get("session_allowed") or [])
        self.__dumped_state = None

    def allow(self, tool_name: str) -> None:
        self.__allowed.add(tool_name)

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.__allowed


class SessionApprovalsFactory(AgentDependencyFactory):
    # One per agent: the turn loop and the REPL share the chat's approvals.
    __singleton__ = True

    @classmethod
    def create(cls) -> SessionApprovals:
        return SessionApprovals()
