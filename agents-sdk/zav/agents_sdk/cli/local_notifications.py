from typing import Type

from zav.agents_sdk.adapters.notifications import FileNotificationChannelFactory
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry


def setup_local_notifications(
    agent_dependency_registry: Type[AgentDependencyRegistry],
) -> None:
    """Register the file notification channel. Only the local CLI runners call
    this, so deployed services cannot select a channel that writes to disk."""
    agent_dependency_registry.register(FileNotificationChannelFactory)
