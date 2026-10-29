from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Optional

from zav.pydantic_compat import BaseModel

from zav.agents_sdk.domain.agent_dependency import DependencyGroup


class NotificationResult(BaseModel):
    status: Literal["sent", "not_sent"]
    status_description: Optional[str] = None


class NotificationChannel(ABC):
    """A channel that reaches the requesting user"""

    channel_name: ClassVar[str]

    @abstractmethod
    async def notify_user(
        self, notification_title: str, notification_content: str
    ) -> NotificationResult:
        raise NotImplementedError


class NotificationChannelGroup(DependencyGroup[NotificationChannel]):
    __collects__ = NotificationChannel
