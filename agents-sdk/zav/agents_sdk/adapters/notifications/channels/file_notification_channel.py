from datetime import datetime, timezone
from pathlib import Path

from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.notifications.notification_channel import (
    NotificationChannel,
    NotificationResult,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class FileNotificationChannelConfiguration(BaseModel):
    directory: str = "notifications"


class FileNotificationChannel(NotificationChannel):
    """Writes each notification to a markdown file, for local development."""

    channel_name = "file"

    def __init__(self, config: FileNotificationChannelConfiguration):
        self.__directory = Path(config.directory)

    async def notify_user(
        self, notification_title: str, notification_content: str
    ) -> NotificationResult:
        self.__directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.__directory / f"{timestamp}.md"
        path.write_text(f"# {notification_title}\n\n{notification_content}\n")
        return NotificationResult(
            status="sent", status_description=f"Saved to {path.resolve()}."
        )


class FileNotificationChannelFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        file_notification_channel_configuration: FileNotificationChannelConfiguration = (  # noqa: E501
            FileNotificationChannelConfiguration()
        ),
    ) -> FileNotificationChannel:
        return FileNotificationChannel(config=file_notification_channel_configuration)
