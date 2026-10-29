from zav.agents_sdk.adapters.notifications.channels import (
    EmailNotificationChannel,
    EmailNotificationChannelFactory,
    FileNotificationChannel,
    FileNotificationChannelConfiguration,
    FileNotificationChannelFactory,
)
from zav.agents_sdk.adapters.notifications.notification_channel import (
    NotificationChannel,
    NotificationChannelGroup,
    NotificationResult,
)
from zav.agents_sdk.adapters.notifications.notification_service import (
    AbstractNotificationService,
    EmailDeliveryNotConfigured,
    LocalNotificationService,
    LocalNotificationServiceFactory,
    NotificationNotDelivered,
    UserEmailUnavailable,
    UserLookupFailed,
    UserOptedOutOfEmails,
)
from zav.agents_sdk.adapters.notifications.notification_tools_source import (
    NotificationToolsSource,
    NotificationToolsSourceConfiguration,
    NotificationToolsSourceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(LocalNotificationServiceFactory)
AgentDependencyRegistry.register(EmailNotificationChannelFactory)
AgentDependencyRegistry.register(NotificationToolsSourceFactory)

__all__ = [
    "AbstractNotificationService",
    "EmailDeliveryNotConfigured",
    "EmailNotificationChannel",
    "EmailNotificationChannelFactory",
    "FileNotificationChannel",
    "FileNotificationChannelConfiguration",
    "FileNotificationChannelFactory",
    "LocalNotificationService",
    "LocalNotificationServiceFactory",
    "NotificationChannel",
    "NotificationChannelGroup",
    "NotificationNotDelivered",
    "NotificationResult",
    "NotificationToolsSource",
    "NotificationToolsSourceConfiguration",
    "NotificationToolsSourceFactory",
    "UserEmailUnavailable",
    "UserLookupFailed",
    "UserOptedOutOfEmails",
]
