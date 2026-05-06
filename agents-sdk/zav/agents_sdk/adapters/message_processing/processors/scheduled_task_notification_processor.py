from typing import AsyncGenerator, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import (
    ChatCompletionSender,
)
from zav.agents_sdk.adapters.message_processing.message_processor import (
    MessageProcessor,
    StreamItem,
)
from zav.agents_sdk.adapters.notifications import AbstractNotificationService
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class ScheduledTaskNotificationProcessorConfiguration(BaseModel):
    enabled: bool = False
    should_notify_user: bool = False
    zav_fe_url: Optional[str] = None
    chat_note_id: Optional[str] = None
    display_name: Optional[str] = None


class ScheduledTaskNotificationProcessor(MessageProcessor):

    source_name = "scheduled_task_notification"

    def __init__(
        self,
        notification_service: AbstractNotificationService,
        config: ScheduledTaskNotificationProcessorConfiguration,
    ):
        self.enabled = config.enabled
        self.__notification_service = notification_service
        self.__should_notify_user = config.should_notify_user
        self.__zav_fe_url = config.zav_fe_url
        self.__chat_note_id = config.chat_note_id
        self.__display_name = config.display_name

    async def process_stream(
        self, stream: AsyncGenerator[StreamItem, None]
    ) -> AsyncGenerator[StreamItem, None]:
        last_content: Optional[str] = None

        async for response, message in stream:
            completion = response.chat_completion
            if (
                completion is not None
                and completion.sender != ChatCompletionSender.TOOL
                and not completion.function_call_request
                and message.content
            ):
                last_content = message.content

            yield response, message

        if not self.__should_notify_user:
            return
        if not last_content:
            return
        if not self.__zav_fe_url or not self.__chat_note_id:
            return

        try:
            await self.__notification_service.send_agent_notification_email(
                content=last_content,
                zav_fe_url=self.__zav_fe_url,
                chat_note_id=self.__chat_note_id,
                agent_display_name=self.__display_name,
            )
        except Exception:
            logger.exception("Failed to send scheduled task notification email")


class ScheduledTaskNotificationProcessorFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notification_service: AbstractNotificationService,
        scheduled_task_notification_processor_configuration: ScheduledTaskNotificationProcessorConfiguration = ScheduledTaskNotificationProcessorConfiguration(),  # noqa: E501
    ) -> ScheduledTaskNotificationProcessor:
        return ScheduledTaskNotificationProcessor(
            notification_service=notification_service,
            config=scheduled_task_notification_processor_configuration,
        )
