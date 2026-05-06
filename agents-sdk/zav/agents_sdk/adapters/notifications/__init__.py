from zav.agents_sdk.adapters.notifications.notification_service import (
    AbstractNotificationService,
    LocalNotificationService,
    LocalNotificationServiceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(LocalNotificationServiceFactory)

__all__ = [
    "AbstractNotificationService",
    "LocalNotificationService",
    "LocalNotificationServiceFactory",
]
