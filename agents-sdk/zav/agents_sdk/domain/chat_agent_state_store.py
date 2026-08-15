from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

DependencyState = Dict[str, Dict[str, Any]]


class AgentsStorageNotConfiguredError(ValueError):
    """A stateful turn needs agent-state storage but the tenant has no
    ``storage_settings.agents``. A ``ValueError`` subclass so the non-streaming
    endpoint still maps it to a 400."""

    def __init__(self, tenant: str) -> None:
        self.tenant = tenant
        super().__init__(f"Agents storage is not configured for tenant {tenant}")


def construct_agent_state_key(
    tenant: str,
    user_uuid: Optional[str],
    session_id: str,
) -> str:
    """Build the relative storage key for a session's agent state.

    Layout: ``agent-state/{tenant}/{user_uuid?}/{session_id}``, appended to the
    agents storage prefix (e.g. ``agent-data``) to give
    ``agent-data/agent-state/{tenant}/.../{session_id}``. ``agent-state`` leads
    so all agent state lives under one namespace within the bucket; the optional
    user segment is omitted when absent, and ``session_id`` (a UUID) keeps each
    conversation uniquely addressable on its own — so the state is keyed by the
    session, not by any index context the chat happened to run in.
    """
    key_parts = ["agent-state", tenant]
    if user_uuid is not None:
        key_parts.append(user_uuid)
    key_parts.append(session_id)
    return "/".join(key_parts)


class ChatAgentStateStore(ABC):
    @abstractmethod
    async def load(
        self,
        tenant: str,
        user_uuid: Optional[str],
        session_id: str,
    ) -> DependencyState:
        raise NotImplementedError

    @abstractmethod
    async def save(
        self,
        tenant: str,
        user_uuid: Optional[str],
        session_id: str,
        state: DependencyState,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        tenant: str,
        user_uuid: Optional[str],
        session_id: str,
    ) -> None:
        """Remove a session's agent state. A no-op if it does not exist."""
        raise NotImplementedError
