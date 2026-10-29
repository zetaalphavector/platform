from typing import Any, Dict, List

from zav.pydantic_compat import Field

from zav.agents_sdk.adapters.notifications.notification_channel import (
    NotificationChannelGroup,
)
from zav.agents_sdk.adapters.tools.tools_source import (
    ToolsSource,
    ToolsSourceConfiguration,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable

DEFAULT_NOTIFICATION_INSTRUCTIONS = """\
Notify the user, when they explicitly ask to receive a notification,
when the answer is ready. Make the notification concise and self-contained so it will
be easily understood by the user out of the chat session."""


class NotificationToolsSourceConfiguration(ToolsSourceConfiguration):
    enabled: bool = False
    channel: str = Field(
        default="email",
        description="channel_name of the notification channel to deliver through.",
    )
    max_notifications_per_turn: int = Field(default=1, gt=0)
    can_subagents_notify: bool = Field(
        default=False,
        description=(
            "Keep notify_user when this agent runs as a sub-agent (started with "
            "task, delegate or a tool's own runs). Each sub-agent then has its own "
            "max_notifications_per_turn."
        ),
    )
    notification_instructions: str = DEFAULT_NOTIFICATION_INSTRUCTIONS


class NotificationToolsSource(ToolsSource):
    """Exposes one `notify_user` tool and routes it to the configured
    notification channel."""

    source_name = "notification_tools"
    source_description = "Notify the user, for example when a scheduled task finishes."

    def __init__(
        self,
        notification_channel_group: NotificationChannelGroup,
        notification_tools_source_configuration: NotificationToolsSourceConfiguration,
        agent_depth: int = 0,
    ):
        self.__channels = {
            channel.channel_name: channel
            for channel in notification_channel_group.items
        }
        self.__config = notification_tools_source_configuration
        self.__notifications_sent = 0
        self.defer_tools = notification_tools_source_configuration.defer_tools
        self.enabled = notification_tools_source_configuration.enabled and (
            agent_depth == 0
            or notification_tools_source_configuration.can_subagents_notify
        )

    async def get_tools(self) -> List[Tool]:
        if not self.enabled:
            return []
        return [Tool.from_callable(name="notify_user", executable=self.notify_user)]

    async def to_prompt(self) -> str:
        return (
            "## Notifying the user\n\n" + self.__config.notification_instructions
            if self.enabled
            else ""
        )

    @streamable(
        running_text="Notifying you...",
        completed_text=(
            "{% if status == 'sent' %}Notification sent."
            "{% else %}Notification not sent.{% endif %}"
        ),
        params_transform=hide,
        response_transform=hide,
    )
    async def notify_user(
        self, notification_title: str, notification_content: str
    ) -> Dict[str, Any]:
        """Send a notification to the current user.

        Args:
            notification_title: A short title.
            notification_content: The message in markdown.

        Returns:
            Dict with 'status': 'sent', or 'not_sent' with a 'status_description'.
        """
        limit = self.__config.max_notifications_per_turn
        if self.__notifications_sent >= limit:
            raise ValueError(f"Notification limit reached: at most {limit} per turn.")
        channel = self.__channels.get(self.__config.channel)
        if channel is None:
            raise ValueError(
                f"Notification channel '{self.__config.channel}' is not available."
            )
        self.__notifications_sent += 1
        result = await channel.notify_user(
            notification_title=notification_title,
            notification_content=notification_content,
        )
        return result.dict(exclude_none=True)


class NotificationToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notification_channel_group: NotificationChannelGroup = NotificationChannelGroup(
            items=[]
        ),
        notification_tools_source_configuration: NotificationToolsSourceConfiguration = (  # noqa: E501
            NotificationToolsSourceConfiguration()
        ),
        agent_depth: int = 0,
    ) -> NotificationToolsSource:
        return NotificationToolsSource(
            notification_channel_group,
            notification_tools_source_configuration,
            agent_depth=agent_depth,
        )
