from typing import Optional

from zav.agents_sdk.adapters.notifications.notification_channel import (
    NotificationChannel,
    NotificationResult,
)
from zav.agents_sdk.adapters.notifications.notification_service import (
    AbstractNotificationService,
    EmailDeliveryNotConfigured,
    UserEmailUnavailable,
    UserLookupFailed,
    UserOptedOutOfEmails,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class EmailNotificationChannel(NotificationChannel):
    channel_name = "email"

    def __init__(
        self,
        notification_service: AbstractNotificationService,
        session_id: Optional[str] = None,
    ):
        self.__notification_service = notification_service
        self.__session_id = session_id

    async def notify_user(
        self, notification_title: str, notification_content: str
    ) -> NotificationResult:
        try:
            await self.__notification_service.send_agent_notification_email(
                subject=notification_title,
                content=notification_content,
                session_id=self.__session_id,
            )
        except UserOptedOutOfEmails:
            return NotificationResult(
                status="not_sent",
                status_description="The user has opted out of email notifications.",
            )
        except UserEmailUnavailable:
            return NotificationResult(
                status="not_sent",
                status_description="The user has no email address.",
            )
        except UserLookupFailed:
            return NotificationResult(
                status="not_sent",
                status_description=(
                    "The user's email address could not be looked up right now; "
                    "try again later."
                ),
            )
        except EmailDeliveryNotConfigured:
            return NotificationResult(
                status="not_sent",
                status_description="Email delivery is not configured.",
            )
        return NotificationResult(
            status="sent",
            status_description=(
                "The email is queued for delivery; it can take a few moments to "
                "arrive."
            ),
        )


class EmailNotificationChannelFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notification_service: AbstractNotificationService,
        session_id: Optional[str] = None,
    ) -> EmailNotificationChannel:
        return EmailNotificationChannel(
            notification_service=notification_service, session_id=session_id
        )
