from abc import ABC, abstractmethod
from typing import Any, Optional

from zav.logging import logger

from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class AbstractNotificationService(ABC):
    @abstractmethod
    async def send_report_notification_email(
        self,
        report_content: str,
        zav_fe_url: str,
        report_uri: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def send_agent_notification_email(
        self,
        content: str,
        zav_fe_url: str,
        chat_note_id: str,
        agent_display_name: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def send_agent_share_notification_email(
        self,
        recipient_uuid: str,
        owner_uuid: str,
        agent_id: str,
        agent_name: str,
        agent_description: Optional[str],
        agent_instructions: Optional[str],
        agent_updated_at: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def send_custom_notification_email(
        self,
        email_type: str,
        recipient_uuid: str,
        recipient_email: str,
        template_payload: dict[str, Any],
        reply_to_email: Optional[str] = None,
        email_tags: Optional[dict[str, str]] = None,
    ) -> None:
        raise NotImplementedError


class LocalNotificationService(AbstractNotificationService):
    async def send_report_notification_email(self, *args, **kwargs) -> None:
        logger.info("Skipping sending notification")

    async def send_agent_notification_email(self, *args, **kwargs) -> None:
        logger.info("Skipping sending notification")

    async def send_agent_share_notification_email(
        self,
        recipient_uuid: str,
        owner_uuid: str,
        agent_id: str,
        agent_name: str,
        agent_description: Optional[str],
        agent_instructions: Optional[str],
        agent_updated_at: str,
    ) -> None:
        logger.info("Skipping sending agent share notification")

    async def send_custom_notification_email(
        self,
        email_type: str,
        recipient_uuid: str,
        recipient_email: str,
        template_payload: dict[str, Any],
        reply_to_email: Optional[str] = None,
        email_tags: Optional[dict[str, str]] = None,
    ) -> None:
        logger.info("Skipping sending custom notification")


class LocalNotificationServiceFactory(AgentDependencyFactory):
    @classmethod
    def create(cls) -> AbstractNotificationService:
        return LocalNotificationService()
